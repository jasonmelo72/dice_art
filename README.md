# Dice Art

A Python project for generating dice art from images. Transform your pictures into unique dice-based representations.

## Features
- Convert images to dice art representations
- Adjustable resolution and dice size
- Support for various image formats
- Custom dice patterns configuration
- Command-line interface

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Pillow (PIL) library
- NumPy library

### Local Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/dice_art.git
cd dice_art
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
```python
from dice_art import DiceArt

# Create a new generator instance
generator = DiceArt()

# Generate dice art from an image
generator.create_from_image("input.jpg", "output.png")
```