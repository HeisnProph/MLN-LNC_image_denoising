"""_summary_
pre process the RGB image to binary image.
May need to adjust threshold value each time for a better result

"""

from PIL import Image

def convert_to_binary_image(image_path, output_path, size=400):
  """
  transfer PNG image to 400*400 binary image

  Args:
      image_path: PNG path
      output_path: out put path
      size: image size
  """

  # open image
  img = Image.open(image_path).convert("L")  # grey
  
  # resizing
  img = img.resize((size, size))

  # convert to binary
  threshold = int(input("Type thresh hold value: ")) # thresh hold
  img = img.point(lambda p: 255 if(p > threshold) else 0)  # 

  # save image
  img.save(output_path)


    
if __name__ == "__main__" :
    
    image_path = input("Insert image path: ") # png path, ex: "data/snake.jpg"
    output_path = input("Insert out put path: ")  # out put path ex: "data/snake_grey.jpg"
    convert_to_binary_image(image_path, output_path)