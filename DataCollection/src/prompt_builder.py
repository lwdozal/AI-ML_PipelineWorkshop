"""
Prompt Builder Module
Constructs prompts for image generation from combined source data
"""

import random
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for synthetic image generation combining multiple data sources.
    """

    def __init__(
        self,
        style: str = "realistic",
        complexity: str = "medium",
        include_temporal: bool = True,
        include_demographics: bool = True,
        themes: Optional[List[str]] = None,
        settings: Optional[List[str]] = None
    ):
        """
        Initialize prompt builder.

        Args:
            style: Image style (realistic, artistic, documentary, photojournalistic)
            complexity: Prompt complexity (simple, medium, complex)
            include_temporal: Include temporal context (time of day, weather, season)
            include_demographics: Include demographic diversity in prompts
            themes: List of social movement themes
            settings: List of settings/locations
        """
        self.style = style
        self.complexity = complexity
        self.include_temporal = include_temporal
        self.include_demographics = include_demographics

        self.themes = themes or [
            "protest", "civic_engagement", "political_rally",
            "community_gathering", "grassroots_organizing", "demonstrations"
        ]

        self.settings = settings or [
            "urban_city_center", "town_square", "university_campus",
            "government_building", "public_park", "street_march"
        ]

        # Temporal elements
        self.times_of_day = ["morning", "midday", "afternoon", "evening", "dusk"]
        self.weather_conditions = ["sunny", "overcast", "partly cloudy", "clear"]
        self.seasons = ["spring", "summer", "fall", "winter"]

        # Visual quality modifiers by style
        self.style_modifiers = {
            "realistic": "photorealistic, high detail, natural lighting",
            "artistic": "artistic composition, dramatic lighting, expressive style",
            "documentary": "documentary photography style, authentic, candid moments",
            "photojournalistic": "photojournalism style, news photography, authentic scene"
        }

    def build_prompt(self, combined_data: Dict[str, Any]) -> str:
        """
        Build image generation prompt from combined data sources.

        Args:
            combined_data: Dictionary with atropia, demographics, and visual_reference data

        Returns:
            Constructed prompt string
        """
        atropia = combined_data['atropia']
        demographics = combined_data['demographics']
        visual_ref = combined_data['visual_reference']

        # Start with base scene description
        prompt_parts = []

        # Add style modifier
        prompt_parts.append(self.style_modifiers.get(self.style, ""))

        # Build scene from visual reference
        scene_description = self._build_scene_description(visual_ref, atropia)
        prompt_parts.append(scene_description)

        # Add demographic context if enabled
        if self.include_demographics:
            demo_context = self._build_demographic_context(demographics)
            if demo_context:
                prompt_parts.append(demo_context)

        # Add location/setting context
        location_context = self._build_location_context(atropia['location'])
        prompt_parts.append(location_context)

        # Add temporal context if enabled
        if self.include_temporal:
            temporal_context = self._build_temporal_context()
            prompt_parts.append(temporal_context)

        # Add complexity-based details
        if self.complexity in ["medium", "complex"]:
            details = self._add_medium_details(atropia)
            prompt_parts.append(details)

        if self.complexity == "complex":
            complex_details = self._add_complex_details(atropia, demographics)
            prompt_parts.append(complex_details)

        # Combine all parts
        prompt = ". ".join([part for part in prompt_parts if part])

        # Add quality and constraint guidelines
        prompt += ". High quality, clear composition, suitable for analysis."

        logger.debug(f"Built prompt with {len(prompt)} characters")

        return prompt

    def _build_scene_description(self, visual_ref: Dict[str, Any], atropia: Dict[str, Any]) -> str:
        """Build base scene description."""
        # Use visual reference description as base
        base_scene = visual_ref.get('description', 'public gathering')

        # Incorporate Atropia theme
        theme = atropia['theme'].replace('_', ' ')

        scene = f"A scene depicting {theme} with {base_scene}"

        return scene

    def _build_demographic_context(self, demographics: Dict[str, Any]) -> str:
        """Build demographic diversity context."""
        age_group = demographics.get('age_group', '')
        occupation = demographics.get('occupation', '')

        contexts = []

        if age_group:
            contexts.append(f"featuring people in the {age_group} age range")

        if occupation and occupation not in ['unemployed']:
            contexts.append(f"including individuals from {occupation} backgrounds")

        if contexts:
            return "Diverse crowd " + ", ".join(contexts)

        return ""

    def _build_location_context(self, location: str) -> str:
        """Build location/setting context."""
        # Clean up location string
        location = location.replace('_', ' ').lower()

        # Determine setting type
        if 'city' in location or 'urban' in location:
            setting_detail = "modern urban environment with buildings and infrastructure"
        elif 'university' in location or 'campus' in location:
            setting_detail = "academic campus setting with institutional architecture"
        elif 'government' in location:
            setting_detail = "governmental district with official buildings"
        elif 'market' in location:
            setting_detail = "commercial district with market atmosphere"
        else:
            setting_detail = "public space with civic architecture"

        return f"Set in {location}, {setting_detail}"

    def _build_temporal_context(self) -> str:
        """Build temporal context (time, weather, season)."""
        time_of_day = random.choice(self.times_of_day)
        weather = random.choice(self.weather_conditions)

        return f"During {time_of_day} with {weather} weather"

    def _add_medium_details(self, atropia: Dict[str, Any]) -> str:
        """Add medium complexity details."""
        # Extract key elements from Atropia excerpt
        excerpt = atropia.get('excerpt', '')

        details = []

        # Add activity indicators
        if 'protest' in excerpt.lower() or 'demonstrate' in excerpt.lower():
            details.append("people actively demonstrating")

        if 'sign' in excerpt.lower() or 'banner' in excerpt.lower():
            details.append("visible signs and banners")

        if 'rally' in excerpt.lower() or 'gather' in excerpt.lower():
            details.append("organized gathering with clear purpose")

        if details:
            return "Scene includes " + ", ".join(details)

        return "Active civic engagement visible in the scene"

    def _add_complex_details(self, atropia: Dict[str, Any], demographics: Dict[str, Any]) -> str:
        """Add complex details for sophisticated prompts."""
        details = []

        # Add socioeconomic context
        setting = demographics.get('setting', 'urban')
        education = demographics.get('education_level', '')

        if setting == 'urban':
            details.append("metropolitan backdrop with diverse architecture")
        elif setting == 'suburban':
            details.append("suburban community setting with residential areas visible")
        else:
            details.append("town setting with local community character")

        # Add crowd composition details
        if education in ['undergraduate', 'graduate', 'professional']:
            details.append("educated populace evident in organized messaging")

        # Add atmospheric details
        details.append("authentic social movement atmosphere with genuine engagement")

        return ". ".join(details)

    def build_batch_prompts(self, combined_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build prompts for a batch of combined data.

        Args:
            combined_data_list: List of combined data dictionaries

        Returns:
            List of dictionaries with prompts and source data
        """
        prompts_data = []

        for i, combined_data in enumerate(combined_data_list):
            prompt = self.build_prompt(combined_data)

            prompt_data = {
                'prompt_id': i + 1,
                'prompt': prompt,
                'source_data': combined_data,
                'style': self.style,
                'complexity': self.complexity
            }

            prompts_data.append(prompt_data)

        logger.info(f"Built {len(prompts_data)} prompts")

        return prompts_data

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that prompt meets quality criteria.

        Args:
            prompt: Prompt string to validate

        Returns:
            True if prompt is valid
        """
        # Check minimum length
        if len(prompt) < 50:
            logger.warning(f"Prompt too short: {len(prompt)} characters")
            return False

        # Check maximum length (Gemini has token limits)
        if len(prompt) > 2000:
            logger.warning(f"Prompt too long: {len(prompt)} characters")
            return False

        # Check for required elements (at least some description)
        if not any(word in prompt.lower() for word in ['scene', 'people', 'crowd', 'gathering', 'protest']):
            logger.warning("Prompt missing key descriptive elements")
            return False

        return True

    def get_prompt_preview(self, combined_data: Dict[str, Any], max_length: int = 200) -> str:
        """
        Get a preview of the prompt (truncated for display).

        Args:
            combined_data: Combined data dictionary
            max_length: Maximum preview length

        Returns:
            Truncated prompt preview
        """
        prompt = self.build_prompt(combined_data)

        if len(prompt) <= max_length:
            return prompt

        return prompt[:max_length] + "..."


class PromptTemplate:
    """
    Template system for different prompt styles.
    """

    @staticmethod
    def social_movement_template(
        theme: str,
        location: str,
        visual_elements: List[str],
        demographic_context: str,
        style: str = "realistic"
    ) -> str:
        """
        Generate prompt from social movement template.

        Args:
            theme: Movement theme
            location: Location/setting
            visual_elements: List of visual elements to include
            demographic_context: Demographic diversity description
            style: Image style

        Returns:
            Formatted prompt
        """
        style_prefix = {
            "realistic": "Photorealistic image of",
            "artistic": "Artistic rendering of",
            "documentary": "Documentary-style photograph of",
            "photojournalistic": "Photojournalism image of"
        }.get(style, "Image of")

        visual_desc = ", ".join(visual_elements)

        prompt = (
            f"{style_prefix} {theme} in {location}. "
            f"{demographic_context}. "
            f"Scene includes {visual_desc}. "
            f"High quality, clear composition."
        )

        return prompt

    @staticmethod
    def minimal_template(theme: str, setting: str) -> str:
        """
        Generate minimal prompt for simple scenarios.

        Args:
            theme: Movement theme
            setting: Location setting

        Returns:
            Minimal prompt
        """
        return f"Photorealistic image of {theme} in {setting}. Clear, high quality."

    @staticmethod
    def detailed_template(
        theme: str,
        location: str,
        demographics: Dict[str, Any],
        temporal: Dict[str, str],
        activities: List[str]
    ) -> str:
        """
        Generate detailed prompt with all context.

        Args:
            theme: Movement theme
            location: Location/setting
            demographics: Demographic information
            temporal: Temporal context (time, weather, season)
            activities: List of activities to depict

        Returns:
            Detailed prompt
        """
        age = demographics.get('age_group', 'diverse ages')
        setting_type = demographics.get('setting', 'urban')

        time_desc = f"{temporal.get('time', 'daytime')} with {temporal.get('weather', 'clear')} weather"
        activities_desc = ", ".join(activities)

        prompt = (
            f"Photorealistic image depicting {theme} in {location}. "
            f"{setting_type.capitalize()} setting during {time_desc}. "
            f"Diverse crowd featuring people of {age}, engaged in {activities_desc}. "
            f"Authentic social movement atmosphere with organized messaging. "
            f"High quality, documentary photography style, clear composition suitable for analysis."
        )

        return prompt
