# Table Detection
This repo was created to help coders that want to extract text from image.


in this project you will find a algorithm to detect table on images and extract structure data to export to csv.

To run, just put an image that you want detect a table and extract information in same folder of "table_detect.py" file.

Open your terminal and type it: "python table_detect.py -i <name_of_image.ext>".

You can go along all the process in your terminal.

The output file has the following structure:
- asset
  - upload
    - image name folder
      - image_name - pre.png (image preprocessed)  
      - image_name - out.png (image with cells and texts detected)
      - cells
        - cells cropped from detected text from image

Feel free to inprove.
