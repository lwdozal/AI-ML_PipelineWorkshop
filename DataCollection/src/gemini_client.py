"""
Gemini API Client Module
Handles image and text generation with rate limiting and retry logic
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import google.generativeai as genai
from PIL import Image
import io

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter with exponential backoff for API requests.
    Tracks requests per minute and per day.
    """

    def __init__(
        self,
        requests_per_minute: int = 10,
        requests_per_day: int = 1000,
        enable_backoff: bool = True,
        initial_delay: float = 2.0,
        backoff_multiplier: float = 2.0,
        max_retries: int = 3
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
            enable_backoff: Enable exponential backoff on errors
            initial_delay: Initial retry delay in seconds
            backoff_multiplier: Multiplier for exponential backoff
            max_retries: Maximum retry attempts
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.enable_backoff = enable_backoff
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_retries = max_retries

        # Track request timestamps
        self.minute_requests: List[datetime] = []
        self.day_requests: List[datetime] = []

    def _clean_old_requests(self):
        """Remove requests older than tracking window."""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_day_ago = now - timedelta(days=1)

        self.minute_requests = [ts for ts in self.minute_requests if ts > one_minute_ago]
        self.day_requests = [ts for ts in self.day_requests if ts > one_day_ago]

    def wait_if_needed(self):
        """Wait if rate limits would be exceeded."""
        self._clean_old_requests()

        # Check daily limit
        if len(self.day_requests) >= self.requests_per_day:
            wait_time = (self.day_requests[0] + timedelta(days=1) - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"Daily rate limit reached. Waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._clean_old_requests()

        # Check minute limit
        if len(self.minute_requests) >= self.requests_per_minute:
            wait_time = (self.minute_requests[0] + timedelta(minutes=1) - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.debug(f"Minute rate limit reached. Waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._clean_old_requests()

    def record_request(self):
        """Record a new request."""
        now = datetime.now()
        self.minute_requests.append(now)
        self.day_requests.append(now)

    def get_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff.

        Args:
            attempt: Current retry attempt (0-indexed)

        Returns:
            Delay in seconds
        """
        if not self.enable_backoff:
            return self.initial_delay

        return self.initial_delay * (self.backoff_multiplier ** attempt)


class GeminiClient:
    """
    Base client for Gemini API interactions.
    """

    def __init__(self, api_key: str, rate_limiter: RateLimiter):
        """
        Initialize Gemini client.

        Args:
            api_key: Google Gemini API key
            rate_limiter: Rate limiter instance
        """
        self.api_key = api_key
        self.rate_limiter = rate_limiter

        # Configure API
        genai.configure(api_key=api_key)

        logger.info("Gemini API client initialized")

    def _make_request_with_retry(self, request_func, *args, **kwargs) -> Any:
        """
        Make API request with retry logic.

        Args:
            request_func: Function to call for request
            *args, **kwargs: Arguments to pass to request function

        Returns:
            Response from API

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(self.rate_limiter.max_retries):
            try:
                # Wait if needed to respect rate limits
                self.rate_limiter.wait_if_needed()

                # Make request
                response = request_func(*args, **kwargs)

                # Record successful request
                self.rate_limiter.record_request()

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.rate_limiter.max_retries}): {e}")

                if attempt < self.rate_limiter.max_retries - 1:
                    delay = self.rate_limiter.get_retry_delay(attempt)
                    logger.info(f"Retrying in {delay:.1f}s")
                    time.sleep(delay)

        # All retries failed
        logger.error(f"All retry attempts failed: {last_error}")
        raise last_error


class GeminiImageGenerator(GeminiClient):
    """
    Generator for synthetic images using Gemini API.
    """

    def __init__(
        self,
        api_key: str,
        rate_limiter: RateLimiter,
        model: str = "gemini-2.5-flash-image",
        resolution: str = "1K",
        aspect_ratio: str = "1:1"
    ):
        """
        Initialize image generator.

        Args:
            api_key: Google Gemini API key
            rate_limiter: Rate limiter instance
            model: Gemini model name
            resolution: Image resolution ("1K" or "4K")
            aspect_ratio: Image aspect ratio
        """
        super().__init__(api_key, rate_limiter)

        self.model_name = model
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio

        # Initialize model
        try:
            self.model = genai.GenerativeModel(model)
            logger.info(f"Initialized {model} for image generation")
        except Exception as e:
            logger.error(f"Failed to initialize model {model}: {e}")
            raise

    def generate_image(self, prompt: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a synthetic image from prompt.

        Args:
            prompt: Text prompt for image generation
            output_path: Optional path to save image

        Returns:
            Dictionary with image data and metadata
        """
        logger.info(f"Generating image with prompt length: {len(prompt)}")

        def _generate():
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
            return response

        # Make request with retry logic
        response = self._make_request_with_retry(_generate)

        # Extract image data
        if not response.parts:
            raise ValueError("No image generated in response")

        # Get image bytes
        image_data = None
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                break

        if not image_data:
            raise ValueError("No image data found in response")

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"Saved image to {output_path}")

        # Prepare metadata
        metadata = {
            'prompt': prompt,
            'model': self.model_name,
            'resolution': self.resolution,
            'aspect_ratio': self.aspect_ratio,
            'generated_at': datetime.now().isoformat(),
            'image_size': image.size,
            'image_mode': image.mode,
        }

        return {
            'image': image,
            'image_data': image_data,
            'metadata': metadata,
            'saved_path': str(output_path) if output_path else None
        }


class GeminiTextGenerator(GeminiClient):
    """
    Generator for text (captions, labels, comments) using Gemini API.
    """

    def __init__(
        self,
        api_key: str,
        rate_limiter: RateLimiter,
        model: str = "gemini-2.5-flash"
    ):
        """
        Initialize text generator.

        Args:
            api_key: Google Gemini API key
            rate_limiter: Rate limiter instance
            model: Gemini model name
        """
        super().__init__(api_key, rate_limiter)

        self.model_name = model

        # Initialize model
        try:
            self.model = genai.GenerativeModel(model)
            logger.info(f"Initialized {model} for text generation")
        except Exception as e:
            logger.error(f"Failed to initialize model {model}: {e}")
            raise

    def generate_caption(self, image: Image.Image, context: Optional[str] = None) -> str:
        """
        Generate descriptive caption for image.

        Args:
            image: PIL Image object
            context: Optional context about the image

        Returns:
            Generated caption
        """
        prompt = "Generate a detailed, descriptive caption for this image. "
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += "Caption should be 1-2 sentences, describing the scene, setting, and main elements."

        def _generate():
            response = self.model.generate_content([prompt, image])
            return response

        response = self._make_request_with_retry(_generate)

        caption = response.text.strip()
        logger.info(f"Generated caption: {caption[:100]}...")

        return caption

    def generate_labels(
        self,
        image: Image.Image,
        categories: List[str],
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate semantic labels for image.

        Args:
            image: PIL Image object
            categories: List of label categories to generate
            context: Optional context about the image

        Returns:
            Dictionary mapping categories to labels
        """
        prompt = f"Analyze this image and provide labels for the following categories: {', '.join(categories)}.\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += "Return as JSON format: {category: label}"

        def _generate():
            response = self.model.generate_content([prompt, image])
            return response

        response = self._make_request_with_retry(_generate)

        # Parse JSON response
        try:
            # Extract JSON from response text
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            labels = json.loads(text)
            logger.info(f"Generated labels: {labels}")
            return labels
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON labels: {e}. Using text response.")
            # Fallback: return raw text
            return {'raw_labels': response.text.strip()}

    def generate_comments(
        self,
        image: Image.Image,
        num_comments: int = 5,
        include_hashtags: bool = True,
        include_emojis: bool = True,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Generate social media-style comments for image.

        Args:
            image: PIL Image object
            num_comments: Number of comments to generate
            include_hashtags: Include hashtags in comments
            include_emojis: Include emojis in comments
            context: Optional context about the image

        Returns:
            List of generated comments
        """
        prompt = f"Generate {num_comments} diverse social media comments for this image. "
        prompt += "Comments should vary in style: supportive, critical, emotional, informational. "

        if include_hashtags:
            prompt += "Include relevant hashtags. "
        if include_emojis:
            prompt += "Include appropriate emojis. "

        if context:
            prompt += f"\n\nContext: {context}\n\n"

        prompt += "\n\nReturn as a JSON array of strings."

        def _generate():
            response = self.model.generate_content([prompt, image])
            return response

        response = self._make_request_with_retry(_generate)

        # Parse JSON response
        try:
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            comments = json.loads(text)
            logger.info(f"Generated {len(comments)} comments")
            return comments
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON comments: {e}. Splitting text response.")
            # Fallback: split by newlines
            comments = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            return comments[:num_comments]


class CheckpointManager:
    """
    Manages checkpoints for resuming interrupted generation.
    """

    def __init__(self, checkpoint_path: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, data: Dict[str, Any]):
        """
        Save checkpoint data.

        Args:
            data: Checkpoint data to save
        """
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data if it exists.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        """Remove checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"Cleared checkpoint at {self.checkpoint_path}")
