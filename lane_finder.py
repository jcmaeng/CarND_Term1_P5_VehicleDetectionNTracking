import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib win32

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


class LaneLineFinder():
    def __init__(self):
        self.leftx_base = None
        self.rightx_base = None
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None
        self.M = None

        self.cal_img = './camera_cal/calibration2.jpg'
        self.lane_img = './test_images/test1.jpg'

        self.is_calibrated = False
        self.skip_win_sliding = False
        self.objp = None
        self.imgp = None

        # set False for video test
        self.view_results = False
        self.save_pipeline_imgs = False

    ##### Functions
    def cal_undistort(self, src_img, obj_points, img_points):
        # Use cv2.calibrateCamera() and cv2.undistort()
        gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)
        undist = cv2.undistort(src_img, mtx, dist, None, mtx)
        return undist

    def find_chessboard_corners(self, src_img, cshape):
        gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        return cv2.findChessboardCorners(gray, cshape, None)

    def warper(self, img, src, dst):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        self.M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def camera_calibration(self, cal_img):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        nx = 9
        ny = 6
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # read image from file
        img = mpimg.imread(cal_img)

        # Find the chessboard corners
        ret, corners = self.find_chessboard_corners(img, (nx,ny))
        # print("ret=", ret)
        # print("corners=", corners)

        if ret is False:
            print("find chessboard corners failed!!")
            return -1, -1

        objpoints.append(objp)
        imgpoints.append(corners)

        undistorted = self.cal_undistort(img, objpoints, imgpoints)

        # cv2.imshow('undist', undistorted)
        # cv2.waitKey()

        # save output result of camera calibration and image undistortion
        cv2.imwrite('./output_images/cal2-undistorted.jpg', undistorted)

        return objpoints, imgpoints

        #### For test - warp the undistorted cal image
        ret, corners2 = self.find_chessboard_corners(undistorted, (nx,ny))
        if ret is False:
            print("find chessboard corners failed!!")
            return -1, -1

        # Draw and display the corners
        undistorted_drawn = undistorted.copy()
        cv2.drawChessboardCorners(undistorted_drawn, (nx,ny), corners2, ret)

        offset = 60
        img_size = (undistorted.shape[1], undistorted.shape[0])
        src_p = np.float32([corners2[0], corners2[nx-1], corners2[-1], corners2[-nx]])
        dst_p = np.float32([[offset,offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
        warped = self.warper(undistorted, src_p, dst_p)

        # save output result of warp perspective
        #cv2.imwrite('./output_images/cal1-warped.jpg', warped)

        #cv2.imshow('warped',warped)
        #cv2.waitKey()

        #cv2.destroyAllWindows()

        return src_p, dst_p

    ###### Finding lane lines

    def do_win_sliding(self, img, nzx, nzy, nwindows=9, margin=100, minpix=50):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)

        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            # (0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            # (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nzy >= win_y_low) & (nzy < win_y_high) & 
            (nzx >= win_xleft_low) &  (nzx < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nzy >= win_y_low) & (nzy < win_y_high) & 
            (nzx >= win_xright_low) &  (nzx < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nzx[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nzx[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return left_lane_inds, right_lane_inds

    def finding_lines(self, warped_bin, skip_win_sliding):
        # Assuming you have created a warped binary image called "warped_bin"

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped_bin, warped_bin, warped_bin))*255
        
        # Choose the number of sliding windows
        nwindows = 9

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_bin.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if self.skip_win_sliding is False:
            left_lane_inds, right_lane_inds = self.do_win_sliding(warped_bin, nonzerox, nonzeroy, nwindows, margin, minpix)
            self.skip_win_sliding = True
        else:
            left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
            self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
            self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))

            right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
            self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
            self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            self.skip_win_sliding = False
            return None
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        ## Visualization
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, warped_bin.shape[0]-1, warped_bin.shape[0] )
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

        window_img = np.zeros_like(out_img)
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return result

    def draw_lane_to_img(self, orig_img, img):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(self.M), (orig_img.shape[1], orig_img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)

        return result

    def calculate_curvature(self):
        # Generate some fake data to represent lane-line pixels
        # self.ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([self.left_fit[0]*y**2 + self.left_fit[1]*y + self.left_fit[2] + np.random.randint(-50, high=51) for y in self.ploty])
        rightx = np.array([self.right_fit[0]*y**2 + self.right_fit[1]*y + self.right_fit[2] + np.random.randint(-50, high=51) for y in self.ploty])

        # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Plot up the fake data
        if self.view_results is True:
            mark_size = 3
            plt.plot(leftx, self.ploty, 'o', color='red', markersize=mark_size)
            plt.plot(rightx, self.ploty, 'o', color='blue', markersize=mark_size)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.plot(self.left_fitx, self.ploty, color='green', linewidth=3)
            plt.plot(self.right_fitx, self.ploty, color='green', linewidth=3)
            plt.gca().invert_yaxis() # to visualize as we do the images
            if self.save_pipeline_imgs is True:
                plt.savefig('./output_images/p7-1_lane_curvature.jpg', format='jpg')
            plt.show()


        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        if self.view_results is True:
            print(left_curverad, right_curverad)
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        if self.view_results is True:
            print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m

        return left_curverad, right_curverad

    def calc_center_diff(self, image):
        #define lane_width with the bottom x points
        lane_width = np.absolute(self.left_fitx[-1] - self.right_fitx[-1])
        #convert pixel to meter 
        xm_per_pix = 3.7/lane_width
        #define center position of the image
        center_pos = (image.shape[1]) * xm_per_pix / 2 
        #define vehicle position from L/R lines
        vehicle_pos = ((self.left_fitx[-1]+ self.right_fitx[-1]) * xm_per_pix ) / 2 
        difference = center_pos - vehicle_pos
        return difference

    ##### Main Procedure
    def process_image(self, img):
        if self.is_calibrated is False:
            self.objp, self.imgp = self.camera_calibration(self.cal_img)
            self.is_calibrated = True

        # limg = mpimg.imread(self.lane_img)
        limg = img.copy()

        # contrast correction for initial image
        limg = cv2.cvtColor(limg, cv2.COLOR_RGB2YUV)
        limg[:,:,0] = cv2.equalizeHist(limg[:,:,0])
        limg = cv2.cvtColor(limg, cv2.COLOR_YUV2RGB)

        limg_undist = self.cal_undistort(limg, self.objp, self.imgp)

        # cv2.imshow('undistorted lane image', limg_undist)
        # cv2.waitKey()

        # Take two channels R and G from undistgorted image for yellow line
        r_channel = limg_undist[:,:,0]
        g_channel = limg_undist[:,:,1]

        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(limg_undist, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Grayscale image
        gray = cv2.cvtColor(limg_undist, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 50
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # # Threshold color channel
        # rg_thresh_min = 180
        # rg_thresh_max = 255
        # rg_binary = np.zeros_like(r_channel|g_channel)
        # rg_binary[((r_channel >= rg_thresh_min) & (r_channel <= rg_thresh_max)) | ((g_channel >= rg_thresh_min) & (g_channel <= rg_thresh_max))] = 1

        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Threshold light channel
        l_thresh_min = 200
        l_thresh_max = 255
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1) | (l_binary == 1)] = 1

        #### warp image
        img_shape = limg_undist.shape
        # src_pts = np.float32([[280,686], [563,484], [786,484], [1111,686]])
        src_pts = np.float32([[280,686], [595,455], [725,455], [1111,686]])
        dst_pts = np.float32([[280, img_shape[0]], [280, 0], [1111, 0], [1111, img_shape[0]]])
        warped = self.warper(combined_binary, src_pts, dst_pts)
        warped_orig = self.warper(limg_undist, src_pts, dst_pts)

        #### finding lane lines with sliding window
        lines_img = self.finding_lines(warped, False)
        if lines_img is None:
            return limg_undist

        #### calculate radius of curvature
        l_curverad, r_curverad = self.calculate_curvature()

        #### draw lane line to image
        result = self.draw_lane_to_img(limg_undist, warped)

        #### add info strings to the result image
        curverad = (l_curverad + r_curverad) / 2
        s = "Radius of Curvature = " + str(curverad)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255,255,255)
        cv2.putText(result, s, (10, 50), font, 1, font_color, 3, cv2.LINE_AA)
        s2 = "Away from Center: " + str(self.calc_center_diff(result))
        cv2.putText(result, s2, (10, 100), font, 1, font_color, 3, cv2.LINE_AA)


        if self.view_results is True:
            # Plotting images
            f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20,10))
            ax1.set_title('Original image')
            ax1.imshow(limg_undist)

            ax2.set_title('HLS converted')
            ax2.imshow(hls)

            ax3.set_title('Stacked thresholds')
            ax3.imshow(color_binary)

            ax4.set_title('Combined S channel and gradient thresholds')
            ax4.imshow(combined_binary, cmap='gray')

            ax5.set_title('Warped original image')
            ax5.imshow(warped_orig)

            ax6.set_title('Warped image')
            ax6.imshow(warped, cmap='gray')

            ax7.set_title('Find lane lines')
            ax7.imshow(lines_img)
            ax7.plot(self.left_fitx, self.ploty, color='yellow')
            ax7.plot(self.right_fitx, self.ploty, color='yellow')
            # ax7.xlim(0, 1280)
            # ax7.ylim(720, 0)

            ax8.set_title('Result')
            ax8.imshow(result)
            # ax8.text(0.5, 0.5, "Radius of Curvature = {}" % ((l_curverad+r_curverad)/2), fontsize=12)


            plt.show()

        if self.save_pipeline_imgs is True:
            output_dir = './output_images'
            mpimg.imsave(output_dir + "/" + "p1_undist_orig.jpg", limg_undist, format='jpg')
            mpimg.imsave(output_dir + "/" + "p2_hls_converted.jpg", hls, format='jpg')
            mpimg.imsave(output_dir + "/" + "p3_stacked_threshold.jpg", color_binary, format='jpg')
            mpimg.imsave(output_dir + "/" + "p4_combined_s_gradient.jpg", combined_binary, cmap='gray', format='jpg')
            mpimg.imsave(output_dir + "/" + "p5_warped_orig.jpg", warped_orig, format='jpg')
            mpimg.imsave(output_dir + "/" + "p6_warped.jpg", warped, cmap='gray', format='jpg')
            mpimg.imsave(output_dir + "/" + "p7_found_lines.jpg", lines_img, format='jpg')
            mpimg.imsave(output_dir + "/" + "p8_result.jpg", result, format='jpg')

        return result

proc = LaneLineFinder()
# proc.process_image(proc.lane_img)
lane_video = './project_video.mp4'
# lane_video = './challenge_video.mp4'
new_clip_output = './output_images/out_project.mp4'

test_clip = VideoFileClip(lane_video, audio=False)
new_clip = test_clip.fl_image(proc.process_image)
new_clip.write_videofile(new_clip_output, audio=False)