import pytest
from src.main import DiceArt

TEST_IMAGE="C:/Users/jason/OneDrive/Pictures/IMG_6622.jpg"

@pytest.fixture(scope="session")
def dice_art_instance():
    dice_art = DiceArt(TEST_IMAGE)
    assert isinstance(dice_art, DiceArt)
    return dice_art

def test_convert_pixel_to_dice_value(dice_art_instance):
    pixel_values = [0, 51, 102, 153, 204, 255]
    expected_dice = [1, 2, 3, 4, 5, 6]
    for pixel, dice in zip(pixel_values, expected_dice):
        assert dice_art_instance._convert_pixel_to_dice(pixel) == dice

@pytest.mark.parametrize("size", [(100,100), (200,200), (500,500)])
def test_resize_image_dimensions(dice_art_instance, size):
    dice_art_instance.resize_image(size)
    assert dice_art_instance.processed_image.size == size

def test_invalid_image_path():
    dice_art = DiceArt()
    with pytest.raises(FileNotFoundError):
        dice_art.load_image("nonexistent.jpg")

def test_convert_to_grayscale(dice_art_instance):
    grey_img = dice_art_instance.convert_to_grayscale()
    # Check if image mode is "L" (grayscale)
    assert grey_img.mode == "L"
