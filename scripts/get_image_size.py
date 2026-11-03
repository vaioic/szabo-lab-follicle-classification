from shared.helper import get_maximum_image_size

max_height, max_width = get_maximum_image_size("../data/Dataset01/cropped_cleaned")

print(f"Max Dimensions: {max_height}, {max_width}")
