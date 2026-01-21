"""
Configuration Management Module
Loads API keys from .env and generation parameters from YAML
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the synthetic data generation pipeline.
    Handles API keys, generation parameters, and output paths.
    """

    def __init__(self, config_dir: Optional[Path] = None, config_file: str = "generation_config.yaml"):
        """
        Initialize configuration from .env and YAML files.

        Args:
            config_dir: Path to config directory (default: DataCollection/config/)
            config_file: Name of YAML config file
        """
        # Determine config directory
        if config_dir is None:
            # Assume we're running from DataCollection/ or its subdirectories
            current_dir = Path.cwd()
            if current_dir.name == "DataCollection":
                self.config_dir = current_dir / "config"
            elif (current_dir.parent.name == "DataCollection"):
                self.config_dir = current_dir.parent / "config"
            else:
                # Search upward for DataCollection directory
                search_dir = current_dir
                while search_dir.name != "DataCollection" and search_dir != search_dir.parent:
                    search_dir = search_dir.parent
                if search_dir.name == "DataCollection":
                    self.config_dir = search_dir / "config"
                else:
                    raise FileNotFoundError("Could not locate DataCollection/config directory")
        else:
            self.config_dir = Path(config_dir)

        # Load environment variables from .env
        env_path = self.config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}. Using environment variables.")

        # Load YAML configuration
        yaml_path = self.config_dir / config_file
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            self._config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {yaml_path}")

        # Validate required configuration
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration sections exist."""
        required_sections = ['generation', 'prompts', 'metadata', 'rate_limiting', 'output']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

    @property
    def api_key(self) -> str:
        """Get Google Gemini API key from environment."""
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        return key

    @property
    def generation(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return self._config['generation']

    @property
    def prompts(self) -> Dict[str, Any]:
        """Get prompt configuration."""
        return self._config['prompts']

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata generation configuration."""
        return self._config['metadata']

    @property
    def rate_limiting(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self._config['rate_limiting']

    @property
    def source_data(self) -> Dict[str, Any]:
        """Get source data configuration."""
        return self._config.get('source_data', {})

    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config['output']

    @property
    def validation(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self._config.get('validation', {})

    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'generation.num_images')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'generation.num_images')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def get_output_path(self, subdir: str = "") -> Path:
        """
        Get absolute path to output directory or subdirectory.

        Args:
            subdir: Subdirectory name (e.g., 'images', 'captions')

        Returns:
            Absolute path to output directory
        """
        # Get base DataCollection directory
        datacoll_dir = self.config_dir.parent

        # Get output directory from config
        output_dir = datacoll_dir / self.output['output_dir']

        if subdir:
            output_dir = output_dir / subdir

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def get_data_path(self, subdir: str = "raw") -> Path:
        """
        Get absolute path to data directory.

        Args:
            subdir: Subdirectory name (e.g., 'raw', 'generated')

        Returns:
            Absolute path to data directory
        """
        datacoll_dir = self.config_dir.parent
        data_dir = datacoll_dir / "data" / subdir

        # Create directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)

        return data_dir

    def save(self, output_path: Optional[Path] = None):
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration (default: overwrite current)
        """
        if output_path is None:
            output_path = self.config_dir / "generation_config.yaml"

        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {output_path}")

    def estimate_cost(self) -> Dict[str, float]:
        """
        Estimate API costs based on configuration.

        Returns:
            Dictionary with cost estimates
        """
        num_images = self.generation['num_images']
        resolution = self.generation['resolution']
        num_comments = self.metadata['num_comments_per_image']

        # Gemini pricing (approximate, as of 2025)
        # Flash model: ~$0.001 per image (1K), ~$0.004 per image (4K)
        # Text generation: ~$0.0001 per request

        if resolution == "1K":
            image_cost_per_unit = 0.001
        else:  # 4K
            image_cost_per_unit = 0.004

        text_cost_per_unit = 0.0001

        # Calculate costs
        image_generation_cost = num_images * image_cost_per_unit
        caption_cost = num_images * text_cost_per_unit
        label_cost = num_images * text_cost_per_unit
        comment_cost = num_images * num_comments * text_cost_per_unit

        total_cost = image_generation_cost + caption_cost + label_cost + comment_cost

        return {
            'num_images': num_images,
            'resolution': resolution,
            'image_generation': round(image_generation_cost, 4),
            'captions': round(caption_cost, 4),
            'labels': round(label_cost, 4),
            'comments': round(comment_cost, 4),
            'total_estimated': round(total_cost, 4),
            'currency': 'USD'
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(images={self.generation['num_images']}, "
            f"model={self.generation['model']}, "
            f"resolution={self.generation['resolution']})"
        )


def setup_logging(config: Config):
    """
    Setup logging based on configuration.

    Args:
        config: Configuration object
    """
    log_config = config.logging_config

    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = []

    # Console handler
    if log_config.get('log_to_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    # File handler
    if log_config.get('log_to_file', True):
        datacoll_dir = config.config_dir.parent
        log_file = datacoll_dir / log_config.get('log_file', 'logs/pipeline.log')
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

    logger.info("Logging configured successfully")


# Convenience function to load configuration
def load_config(config_dir: Optional[Path] = None, config_file: str = "generation_config.yaml") -> Config:
    """
    Load configuration from files.

    Args:
        config_dir: Path to config directory
        config_file: Name of YAML config file

    Returns:
        Configuration object
    """
    config = Config(config_dir, config_file)
    setup_logging(config)
    return config
