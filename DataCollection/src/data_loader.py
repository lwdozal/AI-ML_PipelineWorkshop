"""
Data Loader Module
Fetches and loads source data from Atropia, World Bank, and social media references
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class AtropiaDataLoader:
    """
    Loader for Atropia fictional country data.
    Atropia is a fictional country used in U.S. military training scenarios.
    """

    def __init__(self, data_dir: Path, num_samples: int = 100):
        """
        Initialize Atropia data loader.

        Args:
            data_dir: Directory to save/load data
            num_samples: Number of samples to fetch/load
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.data_file = self.data_dir / "atropia_samples.json"

        # Atropia data source URL
        self.base_url = "https://odin.tradoc.army.mil/DATE/Caucasus/Atropia"

    def fetch_data(self) -> List[Dict[str, Any]]:
        """
        Fetch Atropia news samples.

        Since the actual Atropia website may require authentication or may not be easily scrapable,
        this method generates synthetic Atropia-style data based on typical scenarios.

        Returns:
            List of Atropia news samples
        """
        logger.info("Fetching Atropia data...")

        # Attempt to fetch from real source
        samples = self._try_fetch_from_web()

        # If web fetch fails, generate synthetic samples
        if not samples:
            logger.warning("Could not fetch from web. Generating synthetic Atropia samples.")
            samples = self._generate_synthetic_samples()

        # Save to file
        self._save_data(samples)

        logger.info(f"Loaded {len(samples)} Atropia samples")
        return samples

    def _try_fetch_from_web(self) -> List[Dict[str, Any]]:
        """
        Attempt to fetch Atropia data from web.

        Returns:
            List of samples or empty list if fetch fails
        """
        try:
            # Note: This is a placeholder. The actual Atropia site may require authentication.
            # In a real implementation, you would parse the actual HTML structure.
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Parse news articles (structure depends on actual site)
                # This is a placeholder implementation
                samples = []
                # ... parsing logic would go here
                return samples
            else:
                logger.warning(f"Web fetch returned status {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch from web: {e}")
            return []

    def _generate_synthetic_samples(self) -> List[Dict[str, Any]]:
        """
        Generate synthetic Atropia-style news samples.

        Returns:
            List of synthetic news samples
        """
        themes = [
            "political_unrest",
            "protests",
            "elections",
            "civil_society",
            "government_response",
            "economic_conditions",
            "social_movements"
        ]

        locations = [
            "Atropia City (capital)",
            "Pineland Province",
            "Donovia Border Region",
            "University District",
            "Industrial Zone",
            "Government Quarter",
            "Market District"
        ]

        # Sample news templates
        templates = [
            "Citizens gather in {location} to protest {issue}. Demonstrators call for {demand}.",
            "Government announces new policy regarding {topic}. Civil society groups express {sentiment}.",
            "Large rally held in {location} with estimated {crowd_size} participants. Focus on {issue}.",
            "Opposition party stages demonstration in {location}. Key demands include {demand}.",
            "Community organizers mobilize in {location} around issues of {topic}.",
            "Tensions rise in {location} as protesters clash with authorities over {issue}.",
            "Grassroots movement gains momentum in {location}, advocating for {demand}.",
            "Electoral campaign intensifies in {location} with rallies drawing {crowd_size} supporters.",
            "Student groups organize sit-in at {location} to demand {demand}.",
            "Workers union leads march through {location} protesting {issue}.",
        ]

        issues = [
            "economic reforms",
            "voting rights",
            "government transparency",
            "corruption",
            "labor conditions",
            "education funding",
            "healthcare access",
            "civil liberties",
            "environmental policy",
            "minority rights"
        ]

        demands = [
            "immediate reforms",
            "government accountability",
            "free and fair elections",
            "constitutional changes",
            "policy reversal",
            "institutional transparency",
            "equal representation",
            "economic justice"
        ]

        sentiments = ["strong support", "cautious optimism", "serious concerns", "mixed reactions"]
        topics = issues
        crowd_sizes = ["hundreds of", "thousands of", "several hundred", "over a thousand"]

        samples = []
        for i in range(self.num_samples):
            template = random.choice(templates)
            theme = random.choice(themes)
            location = random.choice(locations)

            # Fill in template
            excerpt = template.format(
                location=location,
                issue=random.choice(issues),
                demand=random.choice(demands),
                sentiment=random.choice(sentiments),
                topic=random.choice(topics),
                crowd_size=random.choice(crowd_sizes)
            )

            sample = {
                'id': f"atropia_{i+1:03d}",
                'title': f"Atropia News: {theme.replace('_', ' ').title()}",
                'excerpt': excerpt,
                'date': f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                'theme': theme,
                'location': location
            }
            samples.append(sample)

        return samples

    def _save_data(self, samples: List[Dict[str, Any]]):
        """Save samples to JSON file."""
        with open(self.data_file, 'w') as f:
            json.dump(samples, f, indent=2)
        logger.info(f"Saved Atropia data to {self.data_file}")

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load Atropia data from file, fetching if necessary.

        Returns:
            List of Atropia samples
        """
        if self.data_file.exists():
            logger.info(f"Loading Atropia data from {self.data_file}")
            with open(self.data_file, 'r') as f:
                return json.load(f)
        else:
            return self.fetch_data()

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get random sample of Atropia data.

        Args:
            n: Number of samples to return

        Returns:
            List of randomly sampled Atropia entries
        """
        data = self.load_data()
        return random.sample(data, min(n, len(data)))


class WorldBankDataLoader:
    """
    Loader for World Bank synthetic demographics data.
    """

    def __init__(self, data_dir: Path, num_profiles: int = 50):
        """
        Initialize World Bank data loader.

        Args:
            data_dir: Directory to save/load data
            num_profiles: Number of demographic profiles to generate
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.num_profiles = num_profiles
        self.data_file = self.data_dir / "worldbank_demographics.csv"

        # World Bank data source URL
        self.data_url = "https://microdata.worldbank.org/index.php/catalog/5906/study-description"

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch World Bank demographics data.

        Since the actual World Bank dataset requires download and may be large,
        this method generates synthetic demographics data following typical patterns.

        Returns:
            DataFrame with demographic profiles
        """
        logger.info("Fetching World Bank demographics...")

        # Generate synthetic demographics
        demographics = self._generate_synthetic_demographics()

        # Save to file
        demographics.to_csv(self.data_file, index=False)
        logger.info(f"Saved {len(demographics)} demographic profiles to {self.data_file}")

        return demographics

    def _generate_synthetic_demographics(self) -> pd.DataFrame:
        """
        Generate synthetic demographic profiles.

        Returns:
            DataFrame with demographic data
        """
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        occupations = [
            "student", "teacher", "healthcare", "retail", "manufacturing",
            "government", "agriculture", "service", "technology", "unemployed"
        ]
        education_levels = ["primary", "secondary", "undergraduate", "graduate", "professional"]
        urban_rural = ["urban", "suburban", "rural"]
        household_sizes = [1, 2, 3, 4, 5, 6]

        profiles = []
        for i in range(self.num_profiles):
            profile = {
                'profile_id': f"wb_{i+1:03d}",
                'age_group': random.choice(age_groups),
                'occupation': random.choice(occupations),
                'education_level': random.choice(education_levels),
                'setting': random.choice(urban_rural),
                'household_size': random.choice(household_sizes),
            }
            profiles.append(profile)

        return pd.DataFrame(profiles)

    def load_data(self) -> pd.DataFrame:
        """
        Load World Bank data from file, fetching if necessary.

        Returns:
            DataFrame with demographic profiles
        """
        if self.data_file.exists():
            logger.info(f"Loading World Bank data from {self.data_file}")
            return pd.read_csv(self.data_file)
        else:
            return self.fetch_data()

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get random sample of demographic profiles.

        Args:
            n: Number of profiles to return

        Returns:
            List of randomly sampled demographic profiles
        """
        data = self.load_data()
        sampled = data.sample(n=min(n, len(data)))
        return sampled.to_dict('records')


class SocialMediaDataLoader:
    """
    Loader for social media reference data (user-provided).
    """

    def __init__(self, data_dir: Path):
        """
        Initialize social media data loader.

        Args:
            data_dir: Directory containing social media reference data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.descriptions_file = self.data_dir / "visual_descriptions.json"
        self.images_dir = self.data_dir / "reference_images"

    def load_descriptions(self) -> List[Dict[str, Any]]:
        """
        Load visual descriptions from JSON file.

        Returns:
            List of visual descriptions
        """
        if not self.descriptions_file.exists():
            logger.warning(f"Descriptions file not found: {self.descriptions_file}")
            return self._generate_placeholder_descriptions()

        logger.info(f"Loading visual descriptions from {self.descriptions_file}")
        with open(self.descriptions_file, 'r') as f:
            return json.load(f)

    def _generate_placeholder_descriptions(self) -> List[Dict[str, Any]]:
        """
        Generate placeholder visual descriptions.

        Returns:
            List of placeholder descriptions
        """
        logger.info("Generating placeholder visual descriptions")

        visual_elements = [
            "crowd of people holding signs",
            "urban street scene with protesters",
            "public gathering in city square",
            "march through downtown area",
            "rally at government building",
            "diverse group of demonstrators",
            "peaceful protest with banners",
            "community organizing event",
            "grassroots mobilization",
            "civic engagement activity"
        ]

        descriptions = []
        for i, element in enumerate(visual_elements):
            descriptions.append({
                'id': f"visual_{i+1:03d}",
                'description': element,
                'setting': random.choice(["urban", "suburban", "town center"]),
                'activity_level': random.choice(["low", "medium", "high"])
            })

        return descriptions

    def sample_description(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get random sample of visual descriptions.

        Args:
            n: Number of descriptions to return

        Returns:
            List of randomly sampled descriptions
        """
        descriptions = self.load_descriptions()
        return random.sample(descriptions, min(n, len(descriptions)))


class DataCombiner:
    """
    Combines data from all three sources for prompt construction.
    """

    def __init__(
        self,
        atropia_loader: AtropiaDataLoader,
        worldbank_loader: WorldBankDataLoader,
        socialmedia_loader: SocialMediaDataLoader
    ):
        """
        Initialize data combiner.

        Args:
            atropia_loader: Atropia data loader
            worldbank_loader: World Bank data loader
            socialmedia_loader: Social media data loader
        """
        self.atropia_loader = atropia_loader
        self.worldbank_loader = worldbank_loader
        self.socialmedia_loader = socialmedia_loader

    def sample_combined(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Sample combined data from all sources.

        Args:
            n: Number of combined samples to return

        Returns:
            List of combined data samples
        """
        combined_samples = []

        for _ in range(n):
            atropia = self.atropia_loader.sample(1)[0]
            worldbank = self.worldbank_loader.sample(1)[0]
            socialmedia = self.socialmedia_loader.sample_description(1)[0]

            combined = {
                'atropia': atropia,
                'demographics': worldbank,
                'visual_reference': socialmedia,
                'sample_id': f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            }
            combined_samples.append(combined)

        return combined_samples
