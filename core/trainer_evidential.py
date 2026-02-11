"""
Evidential trainer for LPVG-EP-DyGAT.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.evidential_classifier import EvidentialLoss, open_set_prediction


class EvidentialTrainer:
    """Trainer for LPVG-EP-DyGAT with evidential loss and OSR metrics."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        method_cfg: Dict[str, Any],
        device: torch.device,
        logger,
        output_dirs: Dict[str, Path]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.method_cfg = method_cfg
        self.device = device
        self.logger = logger
        self.output_dirs = output_dirs

        # Training params
        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay

        # Method params
        loss_cfg = method_cfg.get('loss', {}) if method_cfg else {}
        threshold_cfg = method_cfg.get('threshold', {}) if method_cfg else {}
        self.uncertainty_threshold = threshold_cfg.get('uncertainty', 0.5)
        self.dynamic_threshold = threshold_cfg.get('dynamic', False)
        self.threshold_candidates = int(threshold_cfg.get('candidates', 50))

        # Optimizer
        if config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

        # Scheduler
        if config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.training.step_size,
                gamma=config.training.gamma
            )
        elif config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs
            )
        else:
            self.scheduler = None

        # Loss
        self.criterion = EvidentialLoss(
            num_classes=model.num_classes,
            lambda_kl=loss_cfg.get('lambda_kl', 0.1),
            lambda_proto=loss_cfg.get('lambda_proto', 0.01),
            kl_annealing=loss_cfg.get('kl_annealing', True)
        )

        self.best_h_score = 0.0
        self.best_epoch = 0

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 80)

            train_metrics = self.train_epoch(epoch)

            if epoch % self.config.evaluation.eval_frequency == 0:
                val_metrics = self.evaluate(self.val_loader, update_threshold=True)

                if val_metrics['h_score'] > self.best_h_score:
                    self.best_h_score = val_metrics['h_score']
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ 新的最佳 H-score: {self.best_h_score:.4f}")

                print(
                    f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
                )
                print(
                    f"Val   - Loss: {val_metrics['loss']:.4f}, "
                    f"Known Acc: {val_metrics['known_acc']:.4f}, "
                    f"Unknown Acc: {val_metrics['unknown_acc']:.4f}, "
                    f"H-score: {val_metrics['h_score']:.4f}"
                )

            if self.scheduler is not None:
                self.scheduler.step()

        print(
            f"\n训练完成! 最佳 epoch: {self.best_epoch}, "
            f"最佳 H-score: {self.best_h_score:.4f}"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total = 0
        correct = 0
        loss_edl = 0.0
        loss_kl = 0.0
        loss_proto = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for node_features, edge_index, labels, _ in pbar:
            node_features = [nf.to(self.device) for nf in node_features]
            edge_index = [ei.to(self.device) for ei in edge_index]
            labels = labels.to(self.device)

            known_mask = labels >= 0
            if not known_mask.any():
                continue

            outputs = self.model.forward_batch(node_features, edge_index)

            outputs_known = {
                'alpha': outputs['alpha'][known_mask],
                'S': outputs['S'][known_mask],
                'belief': outputs['belief'][known_mask]
            }
            graph_embeddings = outputs['graph_embedding'][known_mask]
            labels_known = labels[known_mask]

            loss, loss_dict = self.criterion(
                outputs_known,
                graph_embeddings,
                labels_known,
                self.model.get_prototypes(),
                epoch=epoch
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loss_edl += loss_dict.get('loss_edl', 0.0)
            loss_kl += loss_dict.get('loss_kl', 0.0)
            loss_proto += loss_dict.get('loss_proto', 0.0)
            num_batches += 1

            preds = torch.argmax(outputs_known['belief'], dim=1)
            correct += (preds == labels_known).sum().item()
            total += labels_known.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / max(total, 1):.2f}%"
            })

        if num_batches == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'loss_edl': 0.0,
                'loss_kl': 0.0,
                'loss_proto': 0.0
            }

        return {
            'loss': total_loss / num_batches,
            'accuracy': correct / max(total, 1),
            'loss_edl': loss_edl / num_batches,
            'loss_kl': loss_kl / num_batches,
            'loss_proto': loss_proto / num_batches
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, update_threshold: bool = False) -> Dict[str, float]:
        self.model.eval()

        all_preds = []
        all_labels = []
        all_uncertainties = []
        all_known_preds = []
        total_loss = 0.0
        num_batches = 0

        for node_features, edge_index, labels, _ in data_loader:
            node_features = [nf.to(self.device) for nf in node_features]
            edge_index = [ei.to(self.device) for ei in edge_index]
            labels = labels.to(self.device)

            outputs = self.model.forward_batch(node_features, edge_index)

            known_preds = torch.argmax(outputs['belief'], dim=1)
            preds, _ = open_set_prediction(
                outputs,
                uncertainty_threshold=self.uncertainty_threshold
            )

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_uncertainties.append(outputs['uncertainty'].detach().cpu().numpy().reshape(-1))
            all_known_preds.append(known_preds.detach().cpu().numpy().reshape(-1))

            known_mask = labels >= 0
            if known_mask.any():
                outputs_known = {
                    'alpha': outputs['alpha'][known_mask],
                    'S': outputs['S'][known_mask],
                    'belief': outputs['belief'][known_mask]
                }
                graph_embeddings = outputs['graph_embedding'][known_mask]
                labels_known = labels[known_mask]
                loss, _ = self.criterion(
                    outputs_known,
                    graph_embeddings,
                    labels_known,
                    self.model.get_prototypes(),
                    epoch=None
                )
                total_loss += loss.item()
                num_batches += 1

        if len(all_preds) == 0:
            return {
                'loss': 0.0,
                'known_acc': 0.0,
                'unknown_acc': 0.0,
                'h_score': 0.0
            }

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        all_known_preds = np.concatenate(all_known_preds, axis=0)

        known_mask = all_labels >= 0
        unknown_mask = all_labels < 0

        used_tau = self.uncertainty_threshold
        if update_threshold and self.dynamic_threshold and all_uncertainties.size > 0:
            u_min = float(np.min(all_uncertainties))
            u_max = float(np.max(all_uncertainties))
            if u_max > u_min:
                candidates = np.linspace(u_min, u_max, num=max(3, self.threshold_candidates))
                best_h = -1.0
                best_tau = used_tau
                for tau in candidates:
                    preds_tau = all_known_preds.copy()
                    preds_tau[all_uncertainties > tau] = -1

                    if known_mask.sum() > 0:
                        known_acc_tau = (preds_tau[known_mask] == all_labels[known_mask]).mean()
                    else:
                        known_acc_tau = 0.0

                    if unknown_mask.sum() > 0:
                        unknown_acc_tau = (preds_tau[unknown_mask] == -1).mean()
                    else:
                        unknown_acc_tau = 0.0

                    if known_acc_tau + unknown_acc_tau > 0:
                        h_tau = 2 * known_acc_tau * unknown_acc_tau / (known_acc_tau + unknown_acc_tau)
                    else:
                        h_tau = 0.0

                    if h_tau > best_h:
                        best_h = h_tau
                        best_tau = float(tau)

                self.uncertainty_threshold = best_tau
                used_tau = best_tau

        if update_threshold and self.dynamic_threshold:
            all_preds = all_known_preds.copy()
            all_preds[all_uncertainties > used_tau] = -1

        if known_mask.sum() > 0:
            known_acc = (all_preds[known_mask] == all_labels[known_mask]).mean()
        else:
            known_acc = 0.0

        if unknown_mask.sum() > 0:
            unknown_acc = (all_preds[unknown_mask] == -1).mean()
        else:
            unknown_acc = 0.0

        if known_acc + unknown_acc > 0:
            h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        else:
            h_score = 0.0

        avg_loss = total_loss / max(num_batches, 1)

        metrics = {
            'loss': avg_loss,
            'known_acc': float(known_acc),
            'unknown_acc': float(unknown_acc),
            'h_score': float(h_score),
            'used_tau': float(used_tau)
        }

        # Print uncertainty stats for tuning (known vs unknown)
        if all_uncertainties.size > 0:
            known_unc = all_uncertainties[known_mask]
            unknown_unc = all_uncertainties[unknown_mask]

            def _stats(values: np.ndarray) -> Dict[str, float]:
                if values.size == 0:
                    return {
                        'mean': 0.0,
                        'p50': 0.0,
                        'p75': 0.0,
                        'p90': 0.0
                    }
                return {
                    'mean': float(np.mean(values)),
                    'p50': float(np.percentile(values, 50)),
                    'p75': float(np.percentile(values, 75)),
                    'p90': float(np.percentile(values, 90))
                }

            known_stats = _stats(known_unc)
            unknown_stats = _stats(unknown_unc)

            if known_unc.size > 0 and unknown_unc.size > 0:
                suggested_tau = float(0.5 * (known_stats['p75'] + unknown_stats['p50']))
            else:
                suggested_tau = self.uncertainty_threshold

            print(
                "Uncertainty stats | known mean={kmean:.3f} p50={kp50:.3f} p75={kp75:.3f} p90={kp90:.3f} | "
                "unknown mean={umean:.3f} p50={up50:.3f} p75={up75:.3f} p90={up90:.3f} | "
                "suggested_tau={tau:.3f} used_tau={used_tau:.3f}".format(
                    kmean=known_stats['mean'],
                    kp50=known_stats['p50'],
                    kp75=known_stats['p75'],
                    kp90=known_stats['p90'],
                    umean=unknown_stats['mean'],
                    up50=unknown_stats['p50'],
                    up75=unknown_stats['p75'],
                    up90=unknown_stats['p90'],
                    tau=suggested_tau,
                    used_tau=used_tau
                )
            )

            metrics['suggested_tau'] = suggested_tau

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_h_score': self.best_h_score,
            'config': self.config
        }

        checkpoint_path = self.output_dirs['checkpoint_dir'] / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.output_dirs['checkpoint_dir'] / 'best.pth'
            torch.save(checkpoint, best_path)
