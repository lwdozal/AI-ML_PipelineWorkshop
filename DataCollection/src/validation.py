"""
Validation Module
Quality assurance checks for generated images and metadata
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Validator for generated images.
    """

    def __init__(self, min_size_kb: int = 10):
        """
        Initialize image validator.

        Args:
            min_size_kb: Minimum acceptable image file size in KB
        """
        self.min_size_kb = min_size_kb

    def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Validate a single image file.

        Args:
            image_path: Path to image file

        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'file_path': str(image_path)
        }

        # Check file exists
        if not image_path.exists():
            result['valid'] = False
            result['issues'].append("File does not exist")
            return result

        # Check file size
        file_size_kb = image_path.stat().st_size / 1024
        if file_size_kb < self.min_size_kb:
            result['valid'] = False
            result['issues'].append(f"File size too small: {file_size_kb:.2f} KB")

        # Try to open and validate image
        try:
            img = Image.open(image_path)

            # Check image can be loaded
            img.verify()

            # Reopen after verify (verify closes the file)
            img = Image.open(image_path)

            # Check image dimensions
            width, height = img.size
            if width < 100 or height < 100:
                result['valid'] = False
                result['issues'].append(f"Image dimensions too small: {width}x{height}")

            # Check image mode
            if img.mode not in ['RGB', 'RGBA', 'L']:
                result['warnings'].append(f"Unusual image mode: {img.mode}")

            # Check for blank/empty images
            if self._is_blank_image(img):
                result['valid'] = False
                result['issues'].append("Image appears to be blank or empty")

        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"Failed to load image: {str(e)}")

        return result

    def _is_blank_image(self, img: Image.Image, threshold: float = 0.99) -> bool:
        """
        Check if image is blank (all pixels same color).

        Args:
            img: PIL Image
            threshold: Threshold for considering image blank

        Returns:
            True if image appears blank
        """
        try:
            # Convert to numpy array
            img_array = np.array(img)

            # Calculate variance
            if img_array.size == 0:
                return True

            # For RGB images, check variance in each channel
            if len(img_array.shape) == 3:
                variance = np.var(img_array, axis=(0, 1))
                # If all channels have very low variance, likely blank
                if np.all(variance < 10):
                    return True
            else:
                variance = np.var(img_array)
                if variance < 10:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking if image is blank: {e}")
            return False

    def batch_validate(self, image_paths: List[Path]) -> Dict[str, Any]:
        """
        Validate multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            Batch validation results
        """
        results = []
        valid_count = 0
        issue_count = 0

        for image_path in image_paths:
            result = self.validate_image(image_path)
            results.append(result)

            if result['valid']:
                valid_count += 1
            else:
                issue_count += 1

        summary = {
            'total_images': len(image_paths),
            'valid_images': valid_count,
            'invalid_images': issue_count,
            'validation_rate': (valid_count / len(image_paths) * 100) if image_paths else 0,
            'results': results
        }

        return summary


class MetadataValidator:
    """
    Validator for metadata completeness and consistency.
    """

    def __init__(self):
        """Initialize metadata validator."""
        self.required_fields = ['image_id', 'generated_at', 'prompt']

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata structure and completeness.

        Args:
            metadata: Metadata dictionary

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Check required fields
        for field in self.required_fields:
            if field not in metadata:
                result['valid'] = False
                result['issues'].append(f"Missing required field: {field}")

        # Check prompt quality
        if 'prompt' in metadata:
            prompt = metadata['prompt']
            if len(prompt) < 20:
                result['warnings'].append("Prompt seems very short")
            if len(prompt) > 2000:
                result['warnings'].append("Prompt seems very long")

        return result

    def validate_caption(self, caption_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate caption data.

        Args:
            caption_data: Caption dictionary

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        if 'caption' not in caption_data:
            result['valid'] = False
            result['issues'].append("Missing caption field")
            return result

        caption = caption_data['caption']

        # Check caption length
        if len(caption) < 10:
            result['warnings'].append("Caption seems very short")

        if len(caption) > 500:
            result['warnings'].append("Caption seems very long")

        # Check if caption is empty or placeholder
        if not caption or caption.strip() == "":
            result['valid'] = False
            result['issues'].append("Caption is empty")

        return result

    def validate_labels(self, labels_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate labels data.

        Args:
            labels_data: Labels dictionary

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        if 'labels' not in labels_data:
            result['valid'] = False
            result['issues'].append("Missing labels field")
            return result

        labels = labels_data['labels']

        # Check if labels dictionary is empty
        if not labels:
            result['warnings'].append("No labels generated")

        return result

    def validate_comments(self, comments_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate comments data.

        Args:
            comments_data: Comments dictionary

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        if 'comments' not in comments_data:
            result['valid'] = False
            result['issues'].append("Missing comments field")
            return result

        comments = comments_data['comments']

        # Check number of comments
        if not comments:
            result['warnings'].append("No comments generated")

        # Check each comment
        for i, comment in enumerate(comments):
            if not comment or not comment.strip():
                result['warnings'].append(f"Comment {i+1} is empty")

        return result


class DuplicateDetector:
    """
    Detector for duplicate images using perceptual hashing.
    """

    def __init__(self, threshold: int = 95):
        """
        Initialize duplicate detector.

        Args:
            threshold: Similarity threshold (0-100, higher = more strict)
        """
        self.threshold = threshold

    def compute_hash(self, img: Image.Image, hash_size: int = 8) -> str:
        """
        Compute perceptual hash of image.

        Args:
            img: PIL Image
            hash_size: Size of hash

        Returns:
            Hash string
        """
        # Convert to grayscale
        img = img.convert('L')

        # Resize to hash_size x hash_size
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # Convert to numpy array
        pixels = np.array(img).flatten()

        # Compute average
        avg = pixels.mean()

        # Create hash
        hash_bits = (pixels > avg).astype(int)
        hash_string = ''.join(str(bit) for bit in hash_bits)

        return hash_string

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Hamming distance
        """
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity percentage between two hashes.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Similarity percentage (0-100)
        """
        distance = self.hamming_distance(hash1, hash2)
        max_distance = len(hash1)
        similarity = (1 - distance / max_distance) * 100
        return similarity

    def find_duplicates(self, image_paths: List[Path]) -> List[Tuple[Path, Path, float]]:
        """
        Find duplicate images in a list.

        Args:
            image_paths: List of image paths

        Returns:
            List of tuples (image1, image2, similarity)
        """
        logger.info(f"Checking {len(image_paths)} images for duplicates...")

        # Compute hashes
        hashes = {}
        for path in image_paths:
            try:
                img = Image.open(path)
                hash_val = self.compute_hash(img)
                hashes[path] = hash_val
            except Exception as e:
                logger.warning(f"Failed to hash {path}: {e}")

        # Find duplicates
        duplicates = []
        paths = list(hashes.keys())

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                sim = self.similarity(hashes[paths[i]], hashes[paths[j]])
                if sim >= self.threshold:
                    duplicates.append((paths[i], paths[j], sim))

        logger.info(f"Found {len(duplicates)} potential duplicates")

        return duplicates


class DatasetValidator:
    """
    Comprehensive validator for entire dataset.
    """

    def __init__(
        self,
        output_dir: Path,
        min_image_size_kb: int = 10,
        duplicate_threshold: int = 95
    ):
        """
        Initialize dataset validator.

        Args:
            output_dir: Output directory to validate
            min_image_size_kb: Minimum image size in KB
            duplicate_threshold: Duplicate similarity threshold
        """
        self.output_dir = Path(output_dir)
        self.image_validator = ImageValidator(min_image_size_kb)
        self.metadata_validator = MetadataValidator()
        self.duplicate_detector = DuplicateDetector(duplicate_threshold)

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Perform comprehensive dataset validation.

        Returns:
            Validation report
        """
        logger.info("Starting comprehensive dataset validation...")

        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'images': {},
            'metadata': {},
            'captions': {},
            'labels': {},
            'comments': {},
            'duplicates': {},
            'summary': {}
        }

        # Validate images
        images_dir = self.output_dir / "images"
        if images_dir.exists():
            image_paths = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            image_validation = self.image_validator.batch_validate(image_paths)
            report['images'] = image_validation

            # Check for duplicates
            duplicates = self.duplicate_detector.find_duplicates(image_paths)
            report['duplicates'] = {
                'num_duplicates': len(duplicates),
                'duplicate_pairs': [(str(p1), str(p2), sim) for p1, p2, sim in duplicates]
            }

        # Validate metadata completeness
        metadata_dir = self.output_dir / "metadata"
        captions_dir = self.output_dir / "captions"
        labels_dir = self.output_dir / "labels"
        comments_dir = self.output_dir / "comments"

        num_metadata = len(list(metadata_dir.glob("*_metadata.json"))) if metadata_dir.exists() else 0
        num_captions = len(list(captions_dir.glob("*_caption.json"))) if captions_dir.exists() else 0
        num_labels = len(list(labels_dir.glob("*_labels.json"))) if labels_dir.exists() else 0
        num_comments = len(list(comments_dir.glob("*_comments.json"))) if comments_dir.exists() else 0

        num_images = report['images'].get('total_images', 0)

        report['metadata'] = {
            'total_files': num_metadata,
            'coverage_percent': (num_metadata / num_images * 100) if num_images > 0 else 0
        }

        report['captions'] = {
            'total_files': num_captions,
            'coverage_percent': (num_captions / num_images * 100) if num_images > 0 else 0
        }

        report['labels'] = {
            'total_files': num_labels,
            'coverage_percent': (num_labels / num_images * 100) if num_images > 0 else 0
        }

        report['comments'] = {
            'total_files': num_comments,
            'coverage_percent': (num_comments / num_images * 100) if num_images > 0 else 0
        }

        # Overall summary
        valid_images = report['images'].get('valid_images', 0)
        report['summary'] = {
            'total_images': num_images,
            'valid_images': valid_images,
            'validation_rate': (valid_images / num_images * 100) if num_images > 0 else 0,
            'metadata_complete': num_metadata == num_images,
            'captions_complete': num_captions == num_images,
            'labels_complete': num_labels == num_images,
            'comments_complete': num_comments == num_images,
            'duplicates_found': len(duplicates) if 'duplicates' in locals() else 0,
            'dataset_quality': self._assess_quality(report)
        }

        logger.info("Dataset validation complete")

        return report

    def _assess_quality(self, report: Dict[str, Any]) -> str:
        """
        Assess overall dataset quality.

        Args:
            report: Validation report

        Returns:
            Quality assessment string
        """
        validation_rate = report['summary'].get('validation_rate', 0)
        duplicates = report['summary'].get('duplicates_found', 0)

        metadata_complete = report['summary'].get('metadata_complete', False)
        captions_complete = report['summary'].get('captions_complete', False)

        if validation_rate >= 95 and duplicates == 0 and metadata_complete and captions_complete:
            return "Excellent"
        elif validation_rate >= 90 and duplicates <= 2:
            return "Good"
        elif validation_rate >= 80:
            return "Acceptable"
        else:
            return "Needs Improvement"

    def save_report(self, report: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save validation report to file.

        Args:
            report: Validation report
            output_path: Optional output path

        Returns:
            Path to saved report
        """
        if output_path is None:
            qa_dir = self.output_dir.parent / "qa"
            qa_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = qa_dir / f"validation_report_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Saved validation report to {output_path}")

        return output_path
