import cv2
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#Removed below two lines since i am running using python2
# from provided_data.ReadCameraModel import ReadCameraModel
# from provided_data.UndistortImage import UndistortImage
from provided_data.ReadCameraModel import ReadCameraModel
from provided_data.UndistortImage import UndistortImage


def find_images(directory):
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel("%s/provided_data/model" % directory)
    input_dir = "%s/input_images" % directory
    try:
        os.mkdir("%s/undistorted_input_images" % directory)
    except FileExistsError:
        pass
    for tup in os.walk(input_dir):
        root, dirs, files = tup
        root = root.replace("\\", "/")
        files = ["%s/%s" % (root, file) for file in files]
        if len(files) == 0:
            continue
        for file in files:
            read_image = cv2.imread(file, 0)
            color_image = cv2.cvtColor(read_image, cv2.COLOR_BayerGR2BGR)
            undistorted_image = UndistortImage(color_image, LUT)
            cv2.imwrite("%s/undistorted_input_images/%s" % (directory, file.split("/")[-1]),
                        undistorted_image)


def findMatchesBetweenImages(image_1, image_2, num_matches, sift=False):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    sift: bool
        Whether to use SIFT (True) or ORB (False)

    image_1: numpy.array
        The first image

    image_2: numpy.array
        The second image

    num_matches: int
        The number of keypoint matches to find.

    Returns
    -------
    image_1_kp: list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp: list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches: list<cv2.DMatch>
        A list of the top num_matches matches between image_1 and image_2
    """
    feat_detector = eval("cv2.xfeatures2d.SIFT_create()") if sift else cv2.ORB_create(nfeatures=500)
    keypoints_1, description_1 = feat_detector.detectAndCompute(image_1, None)
    keypoints_2, description_2 = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(description_1, description_2),
                     key=lambda x: x.distance)[:num_matches]
    kp1 = np.array([keypoints_1[d_match.queryIdx].pt for d_match in matches])
    kp2 = np.array([keypoints_2[d_match.trainIdx].pt for d_match in matches])
    return kp1, kp2, matches


def show_matches(image_1, image_2, kp1, kp2, reduced=False):
    skip = 2 if reduced else 1
    img1 = image_1[::skip, ::skip, :]
    img2 = image_2[::skip, ::skip, :]
    match_image = np.concatenate((img1, img2), axis=1)
    width = img1.shape[1]
    for k in range(kp1.shape[0]):
        xp1, yp1 = kp1[k].astype(np.int32) // skip
        xp2, yp2 = kp2[k].astype(np.int32) // skip
        cv2.circle(match_image, (xp1, yp1), 2, (0, 0, 255))
        cv2.circle(match_image, (xp2 + width, yp2), 2, (0, 0, 255))
        cv2.line(match_image, (xp1, yp1), (xp2 + width, yp2), (255, 0, 0))
    return match_image


def find_fundamental(kp1, kp2, matches, shape):
    n = -1
    best_F = np.zeros((3, 3), dtype=np.float64)
    xc1 = np.mean(kp1[:, 0])
    yc1 = np.mean(kp1[:, 1])
    xc2 = np.mean(kp2[:, 0])
    yc2 = np.mean(kp2[:, 1])
    sum_xc1 = np.sum(kp1[:, 0])
    sum_yc1 = np.sum(kp1[:, 1])
    sum_xc2 = np.sum(kp2[:, 0])
    sum_yc2 = np.sum(kp2[:, 1])
    kp1_cent = np.stack((kp1[:, 0] - xc1, kp1[:, 1] - yc1), axis=1)
    kp2_cent = np.stack((kp2[:, 0] - xc2, kp2[:, 1] - yc2), axis=1)
    for i in range(50):
        # Pick 8 random unique correspondences
        match_indices = np.arange(len(matches), dtype=np.int32)
        while match_indices.shape[0] > 8:
            match_indices = np.delete(match_indices, np.random.randint(match_indices.shape[0]))

        # Create A matrix
        height, width = shape
        A = np.zeros((0, 9), dtype=np.float64)
        for k in range(kp1.shape[0]):
            xp1 = kp1[k][0] - width / 2
            yp1 = kp1[k][1] - height / 2
            xp2 = kp2[k][0] - width / 2
            yp2 = kp2[k][1] - height / 2
            A = np.append(A, np.array([[xp1*xp2, xp1*yp2, xp1, yp1*xp2, yp1*yp2, yp1, xp2, yp2, 1.0]]), axis=0)

        # Calculate SVD and F matrix
        U, S, Vt = np.linalg.svd(A, full_matrices=True)
        V = np.transpose(Vt)
        F = V[:, 8].reshape((3, 3))
        U, S, Vt = np.linalg.svd(F, full_matrices=True)
        S = np.diag([1.0, 1.0, 0.0])
        F = np.dot(np.dot(U, S), Vt)
        if F[2, 2] == 0.0:
            continue
        F = F / F[2, 2]

        # Normalize F matrix
        rss1 = np.sqrt(np.sum(kp1_cent[:, 0] ** -2 + kp1_cent[:, 1] ** -2) / kp1_cent.shape[0])
        rss2 = np.sqrt(np.sum(kp2_cent[:, 0] ** -2 + kp2_cent[:, 1] ** -2) / kp2_cent.shape[0])
        s1 = np.sqrt(2) / rss1
        s2 = np.sqrt(2) / rss2
        T1 = np.dot(np.diag(np.array([s1, s1, 1.0])),
                    np.array([[1, 0, -sum_xc1 / kp1.shape[0]], [0, 1, -sum_yc1 / kp1.shape[0]], [0, 0, 1]]))
        T2 = np.dot(np.diag(np.array([s2, s2, 1.0])),
                    np.array([[1, 0, -sum_xc2 / kp2.shape[0]], [0, 1, -sum_yc2 / kp2.shape[0]], [0, 0, 1]]))
        F_norm = np.dot(np.dot(T2.transpose(), F), T1)
        F_norm = F_norm / F_norm[2, 2]
        if np.any(np.isnan(F_norm)):
            continue
        local_sum = 0
        for k in range(kp1.shape[0]):
            xp1 = kp1[k][0]
            yp1 = kp1[k][1]
            xp2 = kp2[k][0]
            yp2 = kp2[k][1]
            guess = np.dot(np.array([[xp2, yp2, 1.0]]), F_norm)
            guess = np.abs(np.dot(guess, np.array([[xp1], [yp1], [1.0]]))[0, 0])
            local_sum += guess ** 2
        local_sum = np.sqrt(local_sum)
        if local_sum < n or n < 0:
            n = local_sum
            best_F = F_norm
    return best_F


def find_essential(f_mat, k_mat):
    E = np.dot(np.dot(k_mat.transpose(), f_mat), k_mat)
    U, S, Vt = np.linalg.svd(E)
    S = np.diag(np.array([1.0, 1.0, 0.0]))
    E = np.dot(np.dot(U, S), Vt)
    return E


def camera_pose_estimate(e_mat):
    U, S, Vt = np.linalg.svd(e_mat)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R1 = np.dot(np.dot(U, W), Vt)
    R1 *= np.linalg.det(R1)
    R2 = np.copy(R1)
    R3 = np.dot(np.dot(U, W.transpose()), Vt)
    R3 *= np.linalg.det(R3)
    R4 = np.copy(R3)
    C1 = U[:, 2] * np.linalg.det(R1)
    C2 = -U[:, 2] * np.linalg.det(R2)
    C3 = np.copy(C1)
    C4 = np.copy(C2)
    return [R1, R2, R3, R4], [C1, C2, C3, C4]


def best_camera_pose(r_mat, c_mat, base_pose, inliers_1, inliers_2):
    cp = [np.concatenate((r_mat[i], np.expand_dims(c_mat[i], axis=1)), axis=1) for i in range(4)]
    counts = []
    for i in range(4):
        count = 0
        for j in range(inliers_1.shape[0]):
            p1 = inliers_1[j]
            p2 = inliers_2[j]
            p1_cross = np.array([[0, -1, p1[1]], [1, 0, -p1[0]], [-p1[1], p1[0], 0]])
            p2_cross = np.array([[0, -1, p2[1]], [1, 0, -p2[0]], [-p2[1], p2[0], 0]])
            m_mat = np.vstack([np.dot(p1_cross, base_pose[:3, :]), np.dot(p2_cross, cp[i])])
            U, S, Vt = np.linalg.svd(m_mat)
            point = (Vt[-1] / Vt[-1][3]).reshape((4, 1))[:3]

            # Check if the point is in front of the camera (Cheirality condition)
            r = cp[i][:, :-1]
            t = cp[i][:, -1:]
            if (np.dot(r[2, :], (point - t))) > 0:
                count += 1
        counts.append([count, cp[i]])
    best_pose = counts[int(np.argmax([c[0] for c in counts]))][1]
    return best_pose


def main(argv):
    cwd = os.getcwd().replace("\\", "/")
    parser = argparse.ArgumentParser(description="Project 5:  Visual odometry")
    parser.add_argument('--undistort', action="store_true", help="Undistort images")
    parser.add_argument('--built_in', action="store_true", help="Undistort images")
    args = parser.parse_args(argv)
    undistort = args.undistort
    builtIn = args.built_in
    if undistort:
        find_images(cwd)

    # Look for input images
    input_images_dir = "%s/undistorted_input_images" % cwd
    files = []
    root = ""
    for tup in os.walk(input_images_dir):
        root, dirs, files = tup
        files.sort()
        # files = files[:100]
    sys.stdout.write("\nFound %d input images..." % len(files))
    if len(files) == 0:
        sys.stdout.write("\n")
        exit(0)

    # Extract camera parameters and form camera matrix
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel("%s/provided_data/model" % cwd)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Odometry calculations
    sys.stdout.write("\n")
    base_pose = np.identity(4)
    positions = np.zeros((0, 2), dtype=np.float64)
    trajectory = []
    cv_base_pose = np.identity(4)

    # Read input images
    current_frame = cv2.imread("%s/%s" % (root, files[0]))[::2, ::2, :]
    height = current_frame.shape[0]
    width = current_frame.shape[1]

    H = np.identity(4)

    # for i in range(len(files) - 1):
    for i in range(300, 600, 2):
        # Find matches between two successive frames
        sys.stdout.write("\rProcessing frame %d." % i)
        next_frame = cv2.imread("%s/%s" % (root, files[i + 1]))[::2, ::2, :]

        # Use OpenCV's built-in epipolar geometry calculations
        if builtIn:
            feat_detector = eval("cv2.xfeatures2d.SIFT_create()") if False else cv2.ORB_create(nfeatures=500)
            kp1, dees1 = feat_detector.detectAndCompute(current_frame, None)
            kp2, dees2 = feat_detector.detectAndCompute(next_frame, None)
            bf = cv2.BFMatcher()
            matches = bf.match(dees1, dees2)
            U = []
            V = []
            for m in matches:
                pts_1 = kp1[m.queryIdx]
                x1, y1 = pts_1.pt
                pts_2 = kp2[m.trainIdx]
                x2, y2 = pts_2.pt
                U.append((x1, y1))
                V.append((x2, y2))
            U = np.array(U)
            V = np.array(V)

            E_cv, _ = cv2.findEssentialMat(U, V, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
            _, cv_R, cv_t, mask = cv2.recoverPose(E_cv, U, V, focal=fx, pp=(cx, cy))
            if np.linalg.det(cv_R) < 0:
                cv_R = -cv_R
                cv_t = -cv_t
            new_pose = np.hstack((cv_R, cv_t))
            new_pose = np.vstack((new_pose, np.array([0, 0, 0, 1])))
            x1 = (H[0][3])
            z1 = (H[2][3])
            H = H.dot(new_pose)
            x = (H[0][3])
            z = (H[2][3])
            final_points = np.append(positions, np.array([[x, z]], dtype=np.float64), axis=0)
            # final_points = np.append(final_points, np.array([[x1, x, z1, z]], dtype=np.float64), axis=0)
            # img_current_frame = cv2.resize(img_current_frame, (0,0), fx=0.5, fy=0.5)
            # cv2.imshow('11',img_current_frame)
            # plt.plot([x1, x], [-z1, -z], 'go')
            # plt.pause(0.01)

        else:
            # Find fundamental matrix between two successive frames
            kp1, kp2, matches = findMatchesBetweenImages(current_frame, next_frame, num_matches=30)
            F = find_fundamental(kp1, kp2, matches, (height, width))

            # Find inliers using fundamental matrix
            vector_1 = np.concatenate((kp1, np.ones((kp1.shape[0], 1), dtype=np.float64)), axis=1)
            vector_1 = np.expand_dims(vector_1, axis=2)
            vector_2 = np.concatenate((kp2, np.ones((kp1.shape[0], 1), dtype=np.float64)), axis=1)
            vector_2 = np.expand_dims(vector_2, axis=1)
            distance = np.matmul(np.matmul(vector_2, F), vector_1).squeeze()
            distance_indices = np.where(distance < 0.05)
            inliers_1 = kp1[distance_indices]
            inliers_2 = kp2[distance_indices]

            # Find essential matrix from fundamental matrix and camera calibration matrix
            E = find_essential(F, K)

            # Find best camera pose
            R, C = camera_pose_estimate(E)
            camera_pose = best_camera_pose(R, C, base_pose, inliers_1, inliers_2)
            camera_pose = np.concatenate((camera_pose, np.array([[0, 0, 0, 1]], dtype=np.float)), axis=0)
            base_pose = np.dot(base_pose, camera_pose)
            positions = np.append(positions, np.array([x1, x, z1, z], dtype=np.float64), axis=0)
            # positions = np.append(positions, np.expand_dims(base_pose, axis=0), axis=0)

            # Output position
            # sys.stdout.write("\n")
            # for ii in base_pose:
            #     sys.stdout.write("\n[ ")
            #     for jj in ii:
            #         sys.stdout.write(" %8.4f " % jj)
            #     sys.stdout.write(" ]")
            # sys.stdout.write("\n")
            # match_images = show_matches(current_frame, next_frame, kp1, kp2, reduced=False)
            # if i == 50:
            #     cv2.imwrite("match_image.png", match_images)

        current_frame = next_frame
    # plt.show()

    # Plot camera location using this implementation
    sys.stdout.write("\nProcessed a total of %d images." % len(files))
    fig = plt.figure(figsize=(10.0, 8.0))
    plt_width = int((fig.get_size_inches() * fig.get_dpi())[0])
    plt_height = int((fig.get_size_inches() * fig.get_dpi())[1])
    canvas = FigureCanvas(fig)
    plt.title("Camera locations")
    plt.xlabel("Frame Number (out of %d frames)" % positions.shape[0])
    plt.ylabel("Distance")
    # t_axis = np.arange(positions.shape[0])
    # locations = np.matmul(positions[:, :3, :3], np.expand_dims(positions[:, :3, 3], axis=2))
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    # z_pos = positions[:, 2, 3]
    plt.scatter(x_pos, y_pos, label="Location, built-in plot", s=1.0, c="#0000FF", alpha=0.4)
    plt.legend(loc="best", markerscale=8.0)
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(plt_height, plt_width, 3)
    cv2.imwrite("output_plot.png", plot_image)
    plt.close(fig)

    # Plot camera location using OpenCV built-in functionality
    # cv_fig = plt.figure(figsize=(10.0, 8.0))
    # cv_plt_width = int((cv_fig.get_size_inches() * cv_fig.get_dpi())[0])
    # cv_plt_height = int((cv_fig.get_size_inches() * cv_fig.get_dpi())[1])
    # cv_canvas = FigureCanvas(cv_fig)
    # path_array = np.array(trajectory)
    # plt.title("Camera locations from OpenCV functions")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.scatter(path_array[:, 0], path_array[:, 1], color='r')
    # cv_canvas.draw()
    # cv_plot_image = np.frombuffer(cv_canvas.tostring_rgb(), dtype=np.uint8).reshape(cv_plt_height, cv_plt_width, 3)
    # cv2.imwrite("built_in_plot.png", cv_plot_image)
    # plt.close(cv_fig)

    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
