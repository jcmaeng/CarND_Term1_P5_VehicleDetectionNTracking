import os
import glob
import pickle
import collections

# from lane_finder import *
from car_finder import *

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

### Constants
SAVED_SVC = 'trained_svc.p'
SAVED_SCALER = 'trained_scaler.p'

save_step_results = False
view_results = False
heatmaps = collections.deque(maxlen=4)

# class DriveEnvFinder:
#     def __init__(self):
#         pass

def detect_vehicles(image):
    # lanefinder = LaneFinder()
    carfinder = CarFinder()

    # initialize constants
    scale = 1.5
    color_space = 'YCrCb'
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [400, 656] # Min and max in y to search in slide_window()
    ystart_0 = y_start_stop[0]
    ystop_0 = ystart_0 + 64*2
    ystart_1 = ystart_0
    ystop_1 = y_start_stop[1]
    ystart_2 = ystart_0
    ystop_2 = y_start_stop[1]
    ystarts = [ystart_1, ystart_2]
    ystops = [ystop_1-100, ystop_2]

    if not os.access(SAVED_SVC, os.F_OK) and not os.access(SAVED_SCALER, os.F_OK):
        # Read in cars and notcars
        cars = glob.glob('./training_images/vehicles/**/*.png', recursive=True)
        notcars = glob.glob('./training_images/non-vehicles/**/*.png', recursive=True)
        
        print("# of car images: ", len(cars), "# of not car images: ", len(notcars))

        # Prepare images for training
        sample_size = 8000
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

        ### Training for car detection
        car_features = carfinder.extract_features(cars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = carfinder.extract_features(notcars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)

        print("Features: ", len(car_features), len(notcar_features))
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))



        # Use a linear SVC 
        svc = LinearSVC(C=0.00001)
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        
        print("Training is finished!! Save the model.\n")

        # save the model to file via pickle
        with open(SAVED_SVC, 'wb') as fp_svc:
            pickle.dump(svc, fp_svc)

        with open(SAVED_SCALER, 'wb') as fp_scaler:
            pickle.dump(X_scaler, fp_scaler)

    else:
        with open(SAVED_SVC, 'rb') as fp_svc:
            svc = pickle.load(fp_svc)
        with open(SAVED_SCALER, 'rb') as fp_scaler:
            X_scaler = pickle.load(fp_scaler)
        # print("Load pre-trained SVC and Scaler.\n")


    ### Find cars
    # print("Find Cars")
    scale_list = [1.5, 2.0]
    bbox_list = []
    for scale, ystart, ystop, in zip(scale_list, ystarts, ystops):
        boxes = carfinder.find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
                                    pix_per_cell, cell_per_block, spatial_size, hist_bins)

        bbox_list.extend(boxes)

    img_cars_pos = image.copy()
    for b in bbox_list:
        cv2.rectangle(img_cars_pos, b[0], b[1], (0,0,255), 5)

    ### Make heatmap and remove false positives
    # print("Make heatmap and remove false positives")
    result_image, heatmap = carfinder.get_heatmap_and_boxed_image(image, bbox_list, heatmaps)



    # Save result images of each step to files
    if save_step_results is True:
        output_dir = './output_images'
        mpimg.imsave(output_dir + "/" + "find_cars_by_hog_subsampling.jpg", img_cars_pos, format='jpg')
        mpimg.imsave(output_dir + "/" + "remove_false_positives.jpg", result_image, format='jpg')
        mpimg.imsave(output_dir + "/" + "heatmap.jpg", heatmap, format='jpg')




    if view_results is True:
        f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(20,10))
        ax1.imshow(img_cars_pos)
        ax1.set_title("Car Position by HOG subsampling")

        ax2.imshow(heatmap, cmap='hot')
        ax2.set_title("Heatmap")

        ax3.imshow(result_image)
        ax3.set_title("Remove false positives")

        plt.show()

    return result_image

if __name__ == '__main__':
    # Prepare image
    # image = mpimg.imread('./test_images/test5.jpg')

    # detect_vehicles(image)

    video = './test_video.mp4'
    new_clip_output = './output_images/out_test_cars_finding_only.mp4'

    test_clip = VideoFileClip(video, audio=False)
    new_clip = test_clip.fl_image(detect_vehicles)
    new_clip.write_videofile(new_clip_output, audio=False)