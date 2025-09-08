import numpy as np
import math
from PIL import Image
from typing import Optional

class DiceArt:
    def __init__(self, image_path: Optional[str] = None) -> None:
        """Initialize DiceArt with an image path.

        Args:
            image_path (Optional[str]): Path to the input image file to be processed. 
                          Can be None.

        Returns:
            None

        Attributes:
            original_image: PIL Image object containing the original loaded image
            processed_image: PIL Image object for storing the processed image
            dice_values: Array for storing dice face values after processing
        """
        self.original_image = Image.open(image_path) if image_path else None
        self.processed_image = None
        self.dice_values = None
        self.dice_size = 16  # Default dice size in mm
        self.grid_size = None  # Grid size in (width, height) in dice units
        self.grid_size_inches = None  # Grid size (width, height) in inches

    def create_from_image(self, image_path: str, output_name: str, grid_size_inches: tuple[int, int], dice_size: int = 16) -> None:
        """Convenience method to load and process an image in one step.

        This method combines loading an image from a specified path and processing it
        to generate the dice art representation, saving both preview and mapping files.

        Args:
            image_path (str): The file path to the image to be loaded and processed.
            output_name (str): Base name for the output files (will append _preview.png and _mapping.txt)
            grid_size_inches (tuple[int, int]): The desired (width, height) of the output grid in inches.

        Returns:
            None
        """
        self.load_image(image_path)
        self.calculate_dimensions_from_grid_size(grid_size_inches, dice_size)
        self.process_image()
        self.save_preview(output_name)
        self.save_dice_mapping(output_name)

    def load_image(self, image_path: str) -> None:
        """Load an image from the specified path.

        This method updates the original_image attribute with a new image loaded from the given path.

        Args:
            image_path (str): The file path to the image to be loaded.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified image file does not exist.
        """
        self.original_image = Image.open(image_path)

    def process_image(self, grid_size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """Process the image to create dice art.

        This function converts the original image to grayscale and resizes it to the grid dimensions,
        then maps the pixel brightness values to dice values (1-6). If grid_size is not provided,
        uses the previously set grid_size from instance attributes.

        Args:
            grid_size (Optional[tuple[int, int]], optional): The desired (width, height) of the output grid in dice units.
                                                            If None, uses self.grid_size. Defaults to None.

        Returns:
            np.ndarray: A 2D numpy array containing dice values (1-6) representing the image,
                       where the shape matches the grid dimensions.

        Raises:
            ValueError: If grid_size is None and self.grid_size is None

        Attributes Modified:
            self.processed_image: Stores the resized grayscale image
            self.dice_values: Stores the final mapped dice values array
        """
        # Convert to grayscale
        gray_image = self.convert_to_grayscale()
        
        # Use provided grid_size or fall back to self.grid_size
        grid_size = grid_size if grid_size is not None else self.grid_size
        
        # Resize image to the specified grid size
        if grid_size is None:
            raise ValueError("Grid size must be specified either in process_image() or during initialization")
        self.resize_image(grid_size, gray_image)

        # Convert processed image pixels to dice values
        self.dice_values = self._convert_pixel_to_dice()
        
        return self.dice_values
    
    def convert_to_grayscale(self) -> Image.Image:
        """Convert the original image to grayscale.

        This method converts the original image to grayscale and returns the grayscale version.
        The original image remains unchanged.

        Args:
            None

        Returns:
            Image.Image: Grayscale version of the original image

        Raises:
            ValueError: If no image has been loaded yet (self.original_image is None)
        """
        if self.original_image:
            return self.original_image.convert('L')
        else:
            raise ValueError("No image loaded. Please load an image using load_image() first.")
        
    def resize_image(self, grid_size: tuple[int, int], img: Optional[Image.Image] = None) -> None:
        """Resize the original image to specified dimensions.

        This method resizes the original image to the given grid size using
        high-quality downsampling filter. The resized image is stored in the
        processed_image attribute.

        Args:
            grid_size (tuple[int, int]): The target (width, height) for the resized image.
            img (Optional[Image.Image]): Optional image to resize instead of original image.

        Returns:
            None

        Raises:
            ValueError: If no image has been loaded yet (self.original_image is None)
        """     
        # Choose image source based on input - use provided image if available, otherwise use original image
        image_to_resize = img if img is not None else self.original_image
        if image_to_resize:
            # Resize the image using high-quality downsampling filter
            self.processed_image = image_to_resize.resize(grid_size, Image.Resampling.LANCZOS)
        else:
            raise ValueError("No image loaded. Please load an image using load_image() first.")
        
    def _convert_pixel_to_dice(self, input_array: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert a pixel brightness value (0-255) to a dice face value (1-6).
        This private method maps a given pixel brightness value to a corresponding
        dice face value based on defined brightness ranges.

        Args:
            pixel_value (int): The brightness value of the pixel (0-255).

        Returns:
            int: The corresponding dice face value (1-6).

        Raises:
            ValueError: If pixel_value is not in the range 0-255.
        """
        if input_array is not None:
            # Use provided array
            image_array = input_array
        elif self.processed_image is not None:
            # Convert processed image to numpy array
            image_array = np.array(self.processed_image)
        else:
            raise ValueError("No image data available. Process an image first.")
        # Validate pixel values
        if np.any((image_array < 0) | (image_array > 255)):
            raise ValueError("Pixel values must be in the range 0-255.")
        # Map brightness values (0-255) to dice values (1-6)
        dice_values = np.ceil(image_array / 255 * 6).astype(int)
        # Ensure minimum value is 1
        dice_values = np.clip(dice_values, 1, 6)
        return dice_values

    def save_preview(self, output_path: str) -> None:
        """Save the pixelated preview image.

        This method saves the processed pixelated image to the specified output path.
        Only works if an image has been previously processed.

        Args:
            output_path (str): The file path where the processed image will be saved.
                Should include the filename and extension (e.g., 'output.png')

        Returns:
            None

        Raises:
            AttributeError: If no image has been processed yet (self.processed_image is None)
        """
        if self.grid_size is None:
            raise ValueError("Grid size must be set before saving preview. Process an image first.")
        output_path = f"/res/img/{output_path}_{self.grid_size[0]}x{self.grid_size[1]}in_{self.dice_size}mm_preview.png"
        if self.processed_image:
            self.processed_image.save(output_path)

    def save_dice_mapping(self, output_path: str) -> None:
        """Save the dice values mapping to a text file.

        This function saves the calculated dice mapping values to a text file
        using numpy's savetxt function. The values are saved as integers with
        space delimiter.

        Args:
            output_path (str): The file path where the dice mapping will be saved

        Returns:
            None

        Raises:
            RuntimeError: If dice_values has not been calculated yet (is None)
        """
        if self.grid_size is None:
            raise ValueError("Grid size must be set before saving preview. Process an image first.")
        output_path = f"/res/map/{output_path}_{self.grid_size[0]}x{self.grid_size[1]}in_{self.dice_size}mm_mapping.txt"
        if self.dice_values is not None:
            #np.savetxt(output_path, self.dice_values, fmt='%d', delimiter=' ')
            dice_grid = self.get_dice_grid(show_coordinates=True)
            with open(output_path, 'w') as f:
                f.write(dice_grid)

    def get_dice_grid(self, show_coordinates: bool = True) -> str:
        """Return the dice values as a formatted string grid with optional coordinates.

        This method converts the stored dice values into a string representation where each row
        of dice values is separated by newlines and each value within a row is separated by spaces.
        When show_coordinates is True, it includes row and column numbers.

        Args:
            show_coordinates (bool): Whether to include row and column numbers (default: True)

        Returns:
            str: A formatted string where:
                - Each row of dice values is on a new line
                - Values within each row are space-separated
                - Column numbers are shown at the top (if show_coordinates is True)
                - Row numbers are shown at the left (if show_coordinates is True)
                - Returns empty string if no dice values are stored

        Example:
            If dice_values is [[1, 2], [3, 4]], returns with coordinates:
            "  1 2
             1 1 2
             2 3 4"
        """
        if self.dice_values is None:
            return ""
        
        rows, cols = self.dice_values.shape
        dice_size_in = self.dice_size / 25.4  # Convert mm to inches
        size_covered_in = rows*dice_size_in
        uncovered_area =  self.grid_size_inches[0] - size_covered_in if self.grid_size_inches else 0
        result = [f"Dice Art Grid ({rows}x{cols}) - Total number of dices = {rows*cols} - Uncovered area ~ {uncovered_area:.2f}in on each edge."]
        
        if show_coordinates:
            # Add column numbers header
            header = "    " + " ".join(str(i+1).rjust(2) for i in range(cols))
            result.append(header)
            separator = "  | " + "---"*cols
            result.append(separator)
            
            # Add each row with row number
            for i, row in enumerate(self.dice_values):
                row_str = f"{i+1:2d}| " + " ".join(str(x).rjust(2) for x in row)
                result.append(row_str)
        else:
            # Original format without coordinates
            result = [" ".join(str(x) for x in row) for row in self.dice_values]
            
        return "\n".join(result)

    def calculate_dimensions_from_grid_size(self, grid_size_inches: tuple[int, int], dice_size: int = 16) -> tuple[int, int]:
        """Calculate pixel dimensions based on grid size and cell size.

        This method computes the overall pixel dimensions required to represent
        a grid of dice art based on the specified number of cells in width and height,
        and the size of each cell in pixels.

        Args:
            grid_size (tuple[int, int]): The desired (width, height) of the output grid in inches.
            dice_size (int): Each size of the dice in milimiters (default is 16 mm).

        Returns:
            tuple[int, int]: A tuple containing the calculated (width, height) in pixels.
        """
        self.dice_size = dice_size
        self.grid_size_inches = grid_size_inches
        width, height = self.grid_size_inches
        size_in_inches = self.dice_size / 25.4  # Convert mm to inches
        self.grid_size = (math.trunc(width / size_in_inches), math.trunc(height / size_in_inches))
        return self.grid_size
    
if __name__ == "__main__":
    # Example usage
    dice_art = DiceArt()
    dice_art.create_from_image("res/img/DSC04509.jpg", "output", (12, 12), dice_size=5)