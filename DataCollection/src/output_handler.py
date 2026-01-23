"""
Output Handler Module
Manages saving of generated images, captions, labels, and comments
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Handles organized saving of all generated outputs.
    """

    def __init__(
        self,
        output_dir: Path,
        image_format: str = "png",
        metadata_format: str = "json",
        export_csv: bool = True,
        date_organized: bool = True
    ):
        """
        Initialize output handler.

        Args:
            output_dir: Base output directory
            image_format: Format for saving images (png, jpg)
            metadata_format: Format for metadata (json, csv)
            export_csv: Also export CSV summaries
            date_organized: Organize outputs by date
        """
        self.output_dir = Path(output_dir)
        self.image_format = image_format.lower()
        self.metadata_format = metadata_format.lower()
        self.export_csv = export_csv
        self.date_organized = date_organized

        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.captions_dir = self.output_dir / "captions"
        self.labels_dir = self.output_dir / "labels"
        self.comments_dir = self.output_dir / "comments"
        self.metadata_dir = self.output_dir / "metadata"
        self.analysis_dir = self.output_dir / "analysis"

        self._create_directories()

        # Track saved outputs
        self.saved_count = 0
        self.generation_log = []

    def _create_directories(self):
        """Create all necessary output directories."""
        for directory in [
            self.output_dir,
            self.images_dir,
            self.captions_dir,
            self.labels_dir,
            self.comments_dir,
            self.metadata_dir,
            self.analysis_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directories created at {self.output_dir}")

    def _get_base_filename(self, index: int) -> str:
        """
        Generate base filename for outputs.

        Args:
            index: Index number for this output

        Returns:
            Base filename (without extension)
        """
        if self.date_organized:
            date_str = datetime.now().strftime("%Y%m%d")
            return f"sm_{date_str}_{index:04d}"
        else:
            return f"sm_{index:04d}"

    def save_image(
        self,
        image: Image.Image,
        index: int,
        prompt_data: Dict[str, Any]
    ) -> Path:
        """
        Save generated image with metadata.

        Args:
            image: PIL Image object
            index: Index number for this image
            prompt_data: Prompt and source data used to generate image

        Returns:
            Path to saved image
        """
        base_filename = self._get_base_filename(index)
        image_filename = f"{base_filename}.{self.image_format}"
        image_path = self.images_dir / image_filename

        # Save image
        image.save(image_path, format=self.image_format.upper())
        logger.info(f"Saved image to {image_path}")

        # Save image metadata
        metadata = {
            'image_id': base_filename,
            'image_filename': image_filename,
            'image_path': str(image_path),
            'image_size': image.size,
            'image_mode': image.mode,
            'image_format': self.image_format,
            'generated_at': datetime.now().isoformat(),
            'prompt': prompt_data.get('prompt', ''),
            'source_data': prompt_data.get('source_data', {}),
            'generation_config': {
                'style': prompt_data.get('style', ''),
                'complexity': prompt_data.get('complexity', '')
            }
        }

        metadata_filename = f"{base_filename}_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.debug(f"Saved image metadata to {metadata_path}")

        # Log generation
        self.generation_log.append({
            'index': index,
            'image_id': base_filename,
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat()
        })

        self.saved_count += 1

        return image_path

    def save_caption(
        self,
        image_id: str,
        caption: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save generated caption.

        Args:
            image_id: Image identifier
            caption: Generated caption text
            context: Optional context used for generation

        Returns:
            Path to saved caption file
        """
        caption_data = {
            'image_id': image_id,
            'caption': caption,
            'generated_at': datetime.now().isoformat(),
            'context': context or {}
        }

        caption_filename = f"{image_id}_caption.json"
        caption_path = self.captions_dir / caption_filename

        with open(caption_path, 'w') as f:
            json.dump(caption_data, f, indent=2, default=str)

        logger.info(f"Saved caption for {image_id}")

        return caption_path

    def save_labels(
        self,
        image_id: str,
        labels: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save generated labels.

        Args:
            image_id: Image identifier
            labels: Dictionary of labels by category
            context: Optional context used for generation

        Returns:
            Path to saved labels file
        """
        labels_data = {
            'image_id': image_id,
            'labels': labels,
            'generated_at': datetime.now().isoformat(),
            'context': context or {}
        }

        labels_filename = f"{image_id}_labels.json"
        labels_path = self.labels_dir / labels_filename

        with open(labels_path, 'w') as f:
            json.dump(labels_data, f, indent=2, default=str)

        logger.info(f"Saved labels for {image_id}")

        return labels_path

    def save_comments(
        self,
        image_id: str,
        comments: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save generated comments.

        Args:
            image_id: Image identifier
            comments: List of generated comments
            context: Optional context used for generation

        Returns:
            Path to saved comments file
        """
        comments_data = {
            'image_id': image_id,
            'comments': comments,
            'num_comments': len(comments),
            'generated_at': datetime.now().isoformat(),
            'context': context or {}
        }

        comments_filename = f"{image_id}_comments.json"
        comments_path = self.comments_dir / comments_filename

        with open(comments_path, 'w') as f:
            json.dump(comments_data, f, indent=2, default=str)

        logger.info(f"Saved {len(comments)} comments for {image_id}")

        return comments_path

    def export_captions_csv(self) -> Path:
        """
        Export all captions to CSV file.

        Returns:
            Path to CSV file
        """
        if not self.export_csv:
            logger.warning("CSV export disabled in configuration")
            return None

        captions = []

        for caption_file in self.captions_dir.glob("*_caption.json"):
            with open(caption_file, 'r') as f:
                caption_data = json.load(f)
                captions.append({
                    'image_id': caption_data['image_id'],
                    'caption': caption_data['caption'],
                    'generated_at': caption_data['generated_at']
                })

        if not captions:
            logger.warning("No captions to export")
            return None

        df = pd.DataFrame(captions)
        csv_path = self.output_dir / "all_captions.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Exported {len(captions)} captions to {csv_path}")

        return csv_path

    def export_labels_csv(self) -> Path:
        """
        Export all labels to CSV file.

        Returns:
            Path to CSV file
        """
        if not self.export_csv:
            logger.warning("CSV export disabled in configuration")
            return None

        labels_list = []

        for labels_file in self.labels_dir.glob("*_labels.json"):
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
                row = {'image_id': labels_data['image_id']}
                row.update(labels_data['labels'])
                labels_list.append(row)

        if not labels_list:
            logger.warning("No labels to export")
            return None

        df = pd.DataFrame(labels_list)
        csv_path = self.output_dir / "all_labels.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Exported {len(labels_list)} label sets to {csv_path}")

        return csv_path

    def export_comments_csv(self) -> Path:
        """
        Export all comments to CSV file (one row per comment).

        Returns:
            Path to CSV file
        """
        if not self.export_csv:
            logger.warning("CSV export disabled in configuration")
            return None

        all_comments = []

        for comments_file in self.comments_dir.glob("*_comments.json"):
            with open(comments_file, 'r') as f:
                comments_data = json.load(f)
                image_id = comments_data['image_id']
                for i, comment in enumerate(comments_data['comments']):
                    all_comments.append({
                        'image_id': image_id,
                        'comment_index': i + 1,
                        'comment': comment
                    })

        if not all_comments:
            logger.warning("No comments to export")
            return None

        df = pd.DataFrame(all_comments)
        csv_path = self.output_dir / "all_comments.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Exported {len(all_comments)} comments to {csv_path}")

        return csv_path

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of saved outputs.

        Returns:
            Summary dictionary with counts and paths
        """
        num_images = len(list(self.images_dir.glob(f"*.{self.image_format}")))
        num_captions = len(list(self.captions_dir.glob("*_caption.json")))
        num_labels = len(list(self.labels_dir.glob("*_labels.json")))
        num_comments = len(list(self.comments_dir.glob("*_comments.json")))

        summary = {
            'output_directory': str(self.output_dir),
            'images_saved': num_images,
            'captions_saved': num_captions,
            'labels_saved': num_labels,
            'comments_saved': num_comments,
            'image_format': self.image_format,
            'metadata_format': self.metadata_format,
            'csv_exports_enabled': self.export_csv,
            'directories': {
                'images': str(self.images_dir),
                'captions': str(self.captions_dir),
                'labels': str(self.labels_dir),
                'comments': str(self.comments_dir),
                'metadata': str(self.metadata_dir)
            }
        }

        return summary

    def save_generation_log(self) -> Path:
        """
        Save generation log to file.

        Returns:
            Path to log file
        """
        log_path = self.output_dir / "generation_log.json"

        log_data = {
            'total_generated': len(self.generation_log),
            'generation_started': self.generation_log[0]['timestamp'] if self.generation_log else None,
            'generation_completed': self.generation_log[-1]['timestamp'] if self.generation_log else None,
            'log_entries': self.generation_log
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved generation log to {log_path}")

        return log_path

    def load_image(self, image_id: str) -> Optional[Image.Image]:
        """
        Load a saved image by ID.

        Args:
            image_id: Image identifier

        Returns:
            PIL Image object or None if not found
        """
        image_path = self.images_dir / f"{image_id}.{self.image_format}"

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        return Image.open(image_path)

    def load_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for an image.

        Args:
            image_id: Image identifier

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.metadata_dir / f"{image_id}_metadata.json"

        if not metadata_path.exists():
            logger.warning(f"Metadata not found: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def check_completeness(self) -> Dict[str, Any]:
        """
        Check completeness of generated dataset.

        Returns:
            Dictionary with completeness statistics
        """
        num_images = len(list(self.images_dir.glob(f"*.{self.image_format}")))
        num_captions = len(list(self.captions_dir.glob("*_caption.json")))
        num_labels = len(list(self.labels_dir.glob("*_labels.json")))
        num_comments = len(list(self.comments_dir.glob("*_comments.json")))

        completeness = {
            'total_images': num_images,
            'images_with_captions': num_captions,
            'images_with_labels': num_labels,
            'images_with_comments': num_comments,
            'caption_coverage': (num_captions / num_images * 100) if num_images > 0 else 0,
            'label_coverage': (num_labels / num_images * 100) if num_images > 0 else 0,
            'comment_coverage': (num_comments / num_images * 100) if num_images > 0 else 0,
            'fully_complete': num_images > 0 and num_images == num_captions == num_labels == num_comments
        }

        return completeness
