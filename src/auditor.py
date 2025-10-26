# src/auditor.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class CounterfactualAnalyzer:
    """Analyze how predictions change with image perturbations."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def patch_perturbation_analysis(self, image, patch_size=16, perturbation_type="blur"):
        """
        Analyze how predictions change when different patches are perturbed.

        Args:
            image: PIL Image
            patch_size: Size of patches to perturb
            perturbation_type: Type of perturbation ('blur', 'noise', 'blackout', 'gray')

        Returns:
            dict: Analysis results with visualizations
        """
        original_probs, _, original_labels = self._predict_image(image)
        original_top_label = original_labels[0]
        original_confidence = original_probs[0]

        # Get image dimensions
        width, height = image.size

        # Create grid of patches
        patches_x = width // patch_size
        patches_y = height // patch_size

        # Store results
        confidence_changes = []
        prediction_changes = []
        patch_heatmap = np.zeros((patches_y, patches_x))

        for i in range(patches_y):
            for j in range(patches_x):
                # Create perturbed image
                perturbed_img = self._perturb_patch(
                    image.copy(), j, i, patch_size, perturbation_type
                )

                # Get prediction on perturbed image
                perturbed_probs, _, perturbed_labels = self._predict_image(perturbed_img)
                perturbed_confidence = perturbed_probs[0]
                perturbed_label = perturbed_labels[0]

                # Calculate changes
                confidence_change = perturbed_confidence - original_confidence
                prediction_change = 1 if perturbed_label != original_top_label else 0

                confidence_changes.append(confidence_change)
                prediction_changes.append(prediction_change)
                patch_heatmap[i, j] = confidence_change

        # Create visualization
        fig = self._create_counterfactual_visualization(
            image,
            patch_heatmap,
            patch_size,
            original_top_label,
            original_confidence,
            confidence_changes,
            prediction_changes,
        )

        return {
            "figure": fig,
            "patch_heatmap": patch_heatmap,
            "avg_confidence_change": np.mean(confidence_changes),
            "prediction_flip_rate": np.mean(prediction_changes),
            "most_sensitive_patch": np.unravel_index(np.argmin(patch_heatmap), patch_heatmap.shape),
        }

    def _perturb_patch(self, image, patch_x, patch_y, patch_size, perturbation_type):
        """Apply perturbation to a specific patch."""
        left = patch_x * patch_size
        upper = patch_y * patch_size
        right = left + patch_size
        lower = upper + patch_size

        patch_box = (left, upper, right, lower)

        if perturbation_type == "blur":
            # Extract patch, blur it, and paste back
            patch = image.crop(patch_box)
            blurred_patch = patch.filter(ImageFilter.GaussianBlur(5))
            image.paste(blurred_patch, patch_box)

        elif perturbation_type == "blackout":
            # Black out the patch
            draw = ImageDraw.Draw(image)
            draw.rectangle(patch_box, fill="black")

        elif perturbation_type == "gray":
            # Convert patch to grayscale
            patch = image.crop(patch_box)
            gray_patch = patch.convert("L").convert("RGB")
            image.paste(gray_patch, patch_box)

        elif perturbation_type == "noise":
            # Add noise to patch
            patch = np.array(image.crop(patch_box))
            noise = np.random.normal(0, 50, patch.shape).astype(np.uint8)
            noisy_patch = np.clip(patch + noise, 0, 255).astype(np.uint8)
            image.paste(Image.fromarray(noisy_patch), patch_box)

        return image

    def _predict_image(self, image):
        """Helper function to get predictions."""
        from predictor import predict_image

        return predict_image(image, self.model, self.processor, top_k=5)

    def _create_counterfactual_visualization(
        self,
        image,
        patch_heatmap,
        patch_size,
        original_label,
        original_confidence,
        confidence_changes,
        prediction_changes,
    ):
        """Create visualization for counterfactual analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Original image
        ax1.imshow(image)
        ax1.set_title(
            f"Original Image\nPrediction: {original_label} ({original_confidence:.2%})",
            fontweight="bold",
        )
        ax1.axis("off")

        # Patch sensitivity heatmap
        im = ax2.imshow(patch_heatmap, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax2.set_title(
            "Patch Sensitivity Heatmap\n(Confidence Change When Perturbed)", fontweight="bold"
        )
        ax2.set_xlabel("Patch X")
        ax2.set_ylabel("Patch Y")
        plt.colorbar(im, ax=ax2, label="Confidence Change")

        # Add patch grid to original image
        width, height = image.size
        for i in range(patch_heatmap.shape[0]):
            for j in range(patch_heatmap.shape[1]):
                rect = plt.Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size,
                    patch_size,
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                    alpha=0.3,
                )
                ax1.add_patch(rect)

        # Confidence change distribution
        ax3.hist(confidence_changes, bins=20, alpha=0.7, color="skyblue")
        ax3.axvline(0, color="red", linestyle="--", label="No Change")
        ax3.set_xlabel("Confidence Change")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Confidence Changes", fontweight="bold")
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Prediction flip analysis
        flip_rate = np.mean(prediction_changes)
        ax4.bar(["No Flip", "Flip"], [1 - flip_rate, flip_rate], color=["green", "red"])
        ax4.set_ylabel("Proportion")
        ax4.set_title(f"Prediction Flip Rate: {flip_rate:.2%}", fontweight="bold")
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        return fig


class ConfidenceCalibrationAnalyzer:
    """Analyze model calibration and confidence metrics."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def analyze_calibration(self, test_images, test_labels=None, n_bins=10):
        """
        Analyze model calibration using confidence scores.

        Args:
            test_images: List of PIL Images for testing
            test_labels: Optional true labels for accuracy calculation
            n_bins: Number of bins for calibration curve

        Returns:
            dict: Calibration analysis results
        """
        confidences = []
        predictions = []
        max_confidences = []

        # Get predictions and confidences
        for img in test_images:
            probs, indices, labels = self._predict_image(img)
            max_confidences.append(probs[0])
            predictions.append(labels[0])
            confidences.append(probs)

        max_confidences = np.array(max_confidences)

        # Create calibration analysis
        fig = self._create_calibration_visualization(
            max_confidences, test_labels, predictions, n_bins
        )

        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(
            max_confidences, test_labels, predictions
        )

        return {
            "figure": fig,
            "metrics": calibration_metrics,
            "confidence_distribution": max_confidences,
        }

    def _predict_image(self, image):
        """Helper function to get predictions."""
        from predictor import predict_image

        return predict_image(image, self.model, self.processor, top_k=5)

    def _create_calibration_visualization(self, confidences, true_labels, predictions, n_bins):
        """Create calibration visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color="lightblue", edgecolor="black")
        ax1.set_xlabel("Confidence Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Confidence Scores", fontweight="bold")
        ax1.axvline(
            np.mean(confidences),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(confidences):.3f}",
        )
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Reliability diagram (if true labels available)
        if true_labels is not None:
            # Convert to binary correctness
            correct = np.array([pred == true for pred, true in zip(predictions, true_labels)])

            fraction_of_positives, mean_predicted_prob = calibration_curve(
                correct, confidences, n_bins=n_bins, strategy="uniform"
            )

            ax2.plot(mean_predicted_prob, fraction_of_positives, "s-", label="Model")
            ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax2.set_xlabel("Mean Predicted Probability")
            ax2.set_ylabel("Fraction of Positives")
            ax2.set_title("Reliability Diagram", fontweight="bold")
            ax2.legend()
            ax2.grid(alpha=0.3)

            # Calculate ECE
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(confidences, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            ece = 0
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_conf = np.mean(confidences[mask])
                    bin_acc = np.mean(correct[mask])
                    ece += (np.sum(mask) / len(confidences)) * np.abs(bin_acc - bin_conf)

            ax2.text(
                0.1,
                0.9,
                f"ECE: {ece:.3f}",
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        # Confidence vs accuracy (if true labels available)
        if true_labels is not None:
            confidence_bins = np.linspace(0, 1, n_bins + 1)
            bin_accuracies = []
            bin_confidences = []

            for i in range(n_bins):
                mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_acc = np.mean(correct[mask])
                    bin_conf = np.mean(confidences[mask])
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)

            ax3.plot(bin_confidences, bin_accuracies, "o-", label="Model")
            ax3.plot([0, 1], [0, 1], "k--", label="Ideal")
            ax3.set_xlabel("Average Confidence")
            ax3.set_ylabel("Average Accuracy")
            ax3.set_title("Confidence vs Accuracy", fontweight="bold")
            ax3.legend()
            ax3.grid(alpha=0.3)

        # Top-1 vs Top-5 confidence gap
        if len(confidences) > 0 and isinstance(confidences[0], np.ndarray):
            top1_conf = [c[0] for c in confidences]
            top5_conf = [np.sum(c[:5]) for c in confidences]
            confidence_gap = [t1 - (t5 - t1) / 4 for t1, t5 in zip(top1_conf, top5_conf)]

            ax4.hist(confidence_gap, bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
            ax4.set_xlabel("Confidence Gap (Top-1 vs Rest)")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Distribution of Confidence Gaps", fontweight="bold")
            ax4.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def _calculate_calibration_metrics(self, confidences, true_labels, predictions):
        """Calculate calibration metrics."""
        metrics = {
            "mean_confidence": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "overconfident_rate": float(np.mean(confidences > 0.8)),
            "underconfident_rate": float(np.mean(confidences < 0.2)),
        }

        if true_labels is not None:
            correct = np.array([pred == true for pred, true in zip(predictions, true_labels)])
            accuracy = np.mean(correct)
            avg_confidence = np.mean(confidences)

            metrics.update(
                {
                    "accuracy": float(accuracy),
                    "confidence_gap": float(avg_confidence - accuracy),
                    "brier_score": float(brier_score_loss(correct, confidences)),
                }
            )

        return metrics


class BiasDetector:
    """Detect potential biases in model performance across subgroups."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def analyze_subgroup_performance(self, image_subsets, subset_names, true_labels_subsets=None):
        """
        Analyze performance across different subgroups.

        Args:
            image_subsets: List of image subsets for each subgroup
            subset_names: Names for each subgroup
            true_labels_subsets: Optional true labels for each subset

        Returns:
            dict: Bias analysis results
        """
        subgroup_metrics = {}

        for i, (subset, name) in enumerate(zip(image_subsets, subset_names)):
            confidences = []
            predictions = []

            for img in subset:
                probs, indices, labels = self._predict_image(img)
                confidences.append(probs[0])
                predictions.append(labels[0])

            metrics = {
                "mean_confidence": np.mean(confidences),
                "confidence_std": np.std(confidences),
                "sample_size": len(subset),
            }

            # Calculate accuracy if true labels provided
            if true_labels_subsets is not None and i < len(true_labels_subsets):
                true_labels = true_labels_subsets[i]
                correct = [pred == true for pred, true in zip(predictions, true_labels)]
                metrics["accuracy"] = np.mean(correct)
                metrics["error_rate"] = 1 - metrics["accuracy"]

            subgroup_metrics[name] = metrics

        # Create bias analysis visualization
        fig = self._create_bias_visualization(subgroup_metrics, true_labels_subsets is not None)

        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(subgroup_metrics)

        return {
            "figure": fig,
            "subgroup_metrics": subgroup_metrics,
            "fairness_metrics": fairness_metrics,
        }

    def _predict_image(self, image):
        """Helper function to get predictions."""
        from predictor import predict_image

        return predict_image(image, self.model, self.processor, top_k=5)

    def _create_bias_visualization(self, subgroup_metrics, has_accuracy):
        """Create visualization for bias analysis."""
        if has_accuracy:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        subgroups = list(subgroup_metrics.keys())

        # Confidence by subgroup
        confidences = [metrics["mean_confidence"] for metrics in subgroup_metrics.values()]
        ax1.bar(subgroups, confidences, color="lightblue", alpha=0.7)
        ax1.set_ylabel("Mean Confidence")
        ax1.set_title("Mean Confidence by Subgroup", fontweight="bold")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # Add confidence values on bars
        for i, v in enumerate(confidences):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Sample sizes
        sample_sizes = [metrics["sample_size"] for metrics in subgroup_metrics.values()]
        ax2.bar(subgroups, sample_sizes, color="lightgreen", alpha=0.7)
        ax2.set_ylabel("Sample Size")
        ax2.set_title("Sample Size by Subgroup", fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        # Add sample size values on bars
        for i, v in enumerate(sample_sizes):
            ax2.text(i, v + max(sample_sizes) * 0.01, f"{v}", ha="center", va="bottom")

        # Accuracy by subgroup (if available)
        if has_accuracy:
            accuracies = [metrics.get("accuracy", 0) for metrics in subgroup_metrics.values()]
            ax3.bar(subgroups, accuracies, color="lightcoral", alpha=0.7)
            ax3.set_ylabel("Accuracy")
            ax3.set_title("Accuracy by Subgroup", fontweight="bold")
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(axis="y", alpha=0.3)

            # Add accuracy values on bars
            for i, v in enumerate(accuracies):
                ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        return fig

    def _calculate_fairness_metrics(self, subgroup_metrics):
        """Calculate fairness metrics."""
        fairness_metrics = {}

        # Check if we have accuracy metrics
        has_accuracy = all("accuracy" in metrics for metrics in subgroup_metrics.values())

        if has_accuracy and len(subgroup_metrics) >= 2:
            accuracies = [metrics["accuracy"] for metrics in subgroup_metrics.values()]
            confidences = [metrics["mean_confidence"] for metrics in subgroup_metrics.values()]

            fairness_metrics = {
                "accuracy_range": float(max(accuracies) - min(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "confidence_range": float(max(confidences) - min(confidences)),
                "max_accuracy_disparity": float(
                    max(accuracies) / min(accuracies) if min(accuracies) > 0 else float("inf")
                ),
            }

        return fairness_metrics


# Convenience function to create all auditors
def create_auditors(model, processor):
    """Create all auditing analyzers."""
    return {
        "counterfactual": CounterfactualAnalyzer(model, processor),
        "calibration": ConfidenceCalibrationAnalyzer(model, processor),
        "bias": BiasDetector(model, processor),
    }
