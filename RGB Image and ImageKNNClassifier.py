"""
DSC 20 Mid-Quarter Project
Name: Christian Kim, Jonathan Tran
PID:  A16132108, A15967290
"""

# Part 1: RGB Image #
class RGBImage:
    """
    This class is utilized as a blueprint for the image objects within
    the RGB color spaces. It takes in an input "pixels" through the
    constructor.
    """

    def __init__(self, pixels):
        """
        This constructor initializes the pixels attribute within the RGBImage
        object instance. "pixels" is a 3-D matrix that consists of the color
        channel, the row number, and the column number. The channels are
        numbered based on 0 (red), 1 (green), and 2 (blue)
        """
        self.pixels = pixels

    def size(self):
        """
        The method is a getter that outputs the size of the image based on
        number of rows and number of columns.
        """
        return (len(self.pixels[0]), len(self.pixels[0][0]))

    def get_pixels(self):
        """
        This method outputs a deep copy of the pixels attribute. The output
        is the exact same attribute that is passed into the initiator.
        """
        red_channel = []
        for row in range(len(self.pixels[0])):
            temp = []
            for col in self.pixels[0][row]:
                temp.append(col)
            red_channel.append(temp)
        green_channel = []
        for row in range(len(self.pixels[1])):
            temp = []
            for col in self.pixels[1][row]:
                temp.append(col)
            green_channel.append(temp)
        blue_channel = []
        for row in range(len(self.pixels[-1])):
            temp = []
            for col in self.pixels[-1][row]:
                temp.append(col)
            blue_channel.append(temp)
        new_pixels = [red_channel, green_channel, blue_channel]
        return new_pixels

    def copy(self):
        """
        This method outputs a copy of the RGBImage instance by creating a new
        instance via the "get_pixels()" method created earlier. The copy that
        is returned is a new instance.
        """
        copy_of_pixels = self.get_pixels()
        return RGBImage(copy_of_pixels)

    def get_pixel(self, row, col):
        """
        This method takes in two inputs, "row" and "col," and outputs the color
        of the pixel at this position. The result is a tuple that shows the
        intensity of red, intensity of green, and intensity of blue.
        """
        if type(row) != int:
            raise TypeError()
        if type(col) != int:
            raise TypeError()
        num_rows = len(self.pixels[0])
        num_cols = len(self.pixels[0][0])
        if row < 0 or row >= num_rows:
            raise ValueError()
        if col < 0 or col >= num_cols:
            raise ValueError()
        channel_1 = self.pixels[0]
        channel_2 = self.pixels[1]
        channel_3 = self.pixels[-1]
        red_elem = channel_1[row][col]
        green_elem = channel_2[row][col]
        blue_elem = channel_3[row][col]
        return (red_elem, green_elem, blue_elem)
    
    def set_pixel(self, row, col, new_color):
        """
        This method will change the color of the pixel's position given by
        the three inputs, which are the row, the column, and the new color
        (expressed as a tuple with three different intensity colors). If
        the color intensity is -1, then the intensity is not changed.
        """
        if type(row) != int:
            raise TypeError()
        if type(col) != int:
            raise TypeError()
        num_rows = len(self.pixels[0])
        num_cols = len(self.pixels[0][0])
        if row < 0 or row >= num_rows:
            raise ValueError()
        if col < 0 or col >= num_cols:
            raise ValueError()
        red_int = new_color[0]
        green_int = new_color[1]
        blue_int = new_color[-1]
        red_channel = self.pixels[0]
        green_channel = self.pixels[1]
        blue_channel = self.pixels[-1]
        if red_int != -1:
            red_channel[row][col] = red_int
        if green_int != -1:
            green_channel[row][col] = green_int
        if blue_int != -1:
            blue_channel[row][col] = blue_int

# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    This class incorporates several image methods. All methods have a
    decorator called "@staticmethod", which indicates that the functions
    aren't a part of an instance and is called as a class method. All
    methods return a new RGBImage instance without editing the original.
    """

    @staticmethod
    def negate(image):
        """
        This method outputs an inverted version of the image created. The
        values within the pixels are inverted by the function "255 - value".
        """
        new_copy = image.copy()
        copy = new_copy.get_pixels()
        size = 3
        max_val = 255
        return RGBImage([[list(map(lambda elem: max_val - elem, \
            copy[channel][row])) for row in range(len(copy[channel]))] \
            for channel in range(size)])
      
    @staticmethod
    def tint(image, color):
        """
        This method tinges the image using the "color" input. This input is
        expressed as a tuple with three colors based in red, green, and blue,
        all of which are expressed as values. The tint for each color is
        calculated by taking the average of the current color and the
        color value expressed in the "color" tuple.
        """
        new_copy = image.copy()
        copy = new_copy.get_pixels()
        size = 3
        floor_division_by_2 = 2
        return RGBImage([[list(map(lambda elem: (color[channel] + \
            elem) // floor_division_by_2, copy[channel][row])) \
            for row in range(len(copy[channel]))] for channel in range(size)])
            
    @staticmethod
    def clear_channel(image, channel):
        """
        This method clears the channel of the image. The value represented as
        "channel" is the channel where all values will turn to 0.
        """
        blue_channel = 2
        copy = image.copy()
        new_copy = copy.get_pixels()
        if channel == 0:
            red = [list(map(lambda elem: 0, new_copy[0][row])) \
                    for row in range(len(new_copy[0]))]
            green = new_copy[1]
            blue = new_copy[-1]
        elif channel == 1:
            red = new_copy[0]
            green = [list(map(lambda elem: 0, new_copy[1][row])) \
                    for row in range(len(new_copy[1]))]
            blue = new_copy[-1]
        elif channel == blue_channel:
            red = new_copy[0]
            green = new_copy[1]
            blue = [list(map(lambda elem: 0, new_copy[-1][row])) \
                    for row in range(len(new_copy[-1]))]
        return RGBImage([red, green, blue])

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        This method crops the image by taking in "tl_row" and "tl_col", which
        represent the top-left row and column of the image. The "target_size"
        input represents the size of the crop from the "tl_row" and "tl_col"
        inputs. The "target_size" input helps to navigate the bottom-right
        row and column to crop the image.
        """
        copy = image.copy()
        new_copy = copy.get_pixels()
        size = 3
        return RGBImage([[list(map(lambda elem: elem, row[tl_col:(tl_col + \
            target_size[-1])])) for row in new_copy[channel][tl_row:(tl_row + \
            target_size[0])]] for channel in range(size)])
         
    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        This method conducts an algorithm by changing all pixels that match
        the "color" input in the "chroma_image" input to the pixels that
        are found in the same row and column locations in the
        "background_image" input. The output will be a copy.
        """

        if not isinstance(chroma_image, RGBImage):
            raise TypeError()
        if not isinstance(background_image, RGBImage):
            raise TypeError()
        chroma_num_rows = len(chroma_image.pixels[0])
        chroma_num_cols = len(chroma_image.pixels[0][0])
        background_num_rows = len(background_image.pixels[0])
        background_num_cols = len(background_image.pixels[0][0])
        if chroma_num_rows != background_num_rows:
            raise ValueError()
        if chroma_num_cols != background_num_cols:
            raise ValueError()
        copy = chroma_image.copy()
        chroma_pixels = chroma_image.get_pixels()
        background_pixels = background_image.get_pixels()
        red_chroma = chroma_pixels[0]
        green_chroma = chroma_pixels[1]
        blue_chroma = chroma_pixels[-1]
        red_background = background_pixels[0]
        green_background = background_pixels[1]
        blue_background = background_pixels[-1]
        for row in range(len(red_chroma)):
            for col in range(len(red_chroma[0])):
                if color == (red_chroma[row][col], green_chroma[row][col], \
                    blue_chroma[row][col]):
                    new_color = background_image.get_pixel(row, col)
                    copy.set_pixel(row, col, new_color)
        return copy

    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        This method rotates an image by 180 degrees.
        """
        copy = image.copy()
        new_copy = copy.get_pixels()
        red = new_copy[0]
        green = new_copy[1]
        blue = new_copy[-1]
        rotate_red_1 = [list(reversed(column)) for column in zip(*red)]
        rotate_red_2 = [list(reversed(column)) \
                        for column in zip(*rotate_red_1)]
        rotate_green_1 = [list(reversed(column)) for column in zip(*green)]
        rotate_green_2 = [list(reversed(column)) \
                        for column in zip(*rotate_green_1)]
        rotate_blue_1 = [list(reversed(column)) for column in zip(*blue)]
        rotate_blue_2 = [list(reversed(column)) \
                        for column in zip(*rotate_blue_1)]
        return RGBImage([rotate_red_2, rotate_green_2, rotate_blue_2])

# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    This class attempts to make prediction through the pattern of previous
    data. This classifier is trained by using training data within the model
    to predict the most popular labels in a collection of nearest training
    data. One way to do this process if through the use of the Euclidean
    distance, which is described in the method "distance()".
    """

    def __init__(self, n_neighbors):
        """
        This constructor initializes a ImageKNNClassifier instance along with
        an input called "n_neighbors", which displays the size of the nearest
        neighborhood.
        """
        self.n_neighbors = n_neighbors
        self.data = False
        
    def fit(self, data):
        """
        This method will fit a classifier by storing the input "data" into
        the classifier instance. The "data" input is a list containing
        tuples of "image" and "label", where "image" is an RGBImage and
        "label" is a string.
        """
        if len(data) <= self.n_neighbors:
            raise ValueError
        if self.data != False:
            raise ValueError
        self.data = data
        
    @staticmethod
    def distance(image1, image2):
        """
        This method outputs the distance between the inputs "image1" and
        "image2". The distance is calculated by taking the square root sum
        of each position, which contains the channel, row, and column
        """
        if not isinstance(image1, RGBImage):
            raise TypeError()
        if not isinstance(image2, RGBImage):
            raise TypeError()
        img1_num_rows = len(image1.pixels[0])
        img1_num_cols = len(image1.pixels[0][0])
        img2_num_rows = len(image2.pixels[0])
        img2_num_cols = len(image2.pixels[0][0])
        if img1_num_rows != img2_num_rows:
            raise ValueError()
        if img1_num_cols != img2_num_cols:
            raise ValueError()
        red_1 = image1.get_pixels()[0]
        red_2 = image2.get_pixels()[0]
        green_1 = image1.get_pixels()[1]
        green_2 = image2.get_pixels()[1]
        blue_1 = image1.get_pixels()[-1]
        blue_2 = image2.get_pixels()[-1]
        squared = 2
        square_root = 1/2
        red_sum = sum([(red_1[row][col] - red_2[row][col]) ** squared \
            for row in range(len(red_1)) for col in range(len(red_1[0]))])
        green_sum = sum([(green_1[row][col] - green_2[row][col]) ** squared \
            for row in range(len(green_1)) for col in range(len(green_1[0]))])
        blue_sum = sum([(blue_1[row][col] - blue_2[row][col]) ** squared \
            for row in range(len(blue_1)) for col in range(len(blue_1[0]))])
        total_distance = red_sum + green_sum + blue_sum
        euclidean_distance = total_distance ** (square_root)
        return euclidean_distance

    @staticmethod
    def vote(candidates):
        """
        This method selected the most popular label from the input
        "candidates", which is a list of labels of the nearest neighbors.
        If there is a tie, then any of them is returned.
        """
        vote_count = {}
        for elem in candidates:
            if elem not in vote_count:
                vote_count[elem] = 1
            else:
                count = vote_count[elem]
                count += 1
                vote_count[elem] = count
        max_val = max(vote_count.values())
        for key in vote_count.keys():
            if vote_count[key] == max_val:
                return key

    def predict(self, image):
        """
        This method predicts the label of the "image" input by incorporating
        the KNN classification algorithm. The "vote()" method is utilized
        to help with the prediction.
        """
        if self.data == False:
            raise ValueError
        distance_list = [(ImageKNNClassifier.distance(image, img[0]), \
            img[1]) for img in self.data]
        sorted_list = sorted(distance_list)
        top_n = [elem[-1] for elem in sorted_list[:self.n_neighbors]]
        return ImageKNNClassifier.vote(top_n)
