import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import time
from PIL import Image
import math




DEMO_IMAGE = 'demo/demo.jpg'
DEMO_VIDEO = 'demo/demo.mp4'

# Basic App Scaffolding
st.title('Face acupoint detection using FaceMesh mediapipe')

## Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## Create Sidebar
st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('Parameter')

## Define available pages in selectiostreamlit run face_mesh_app.pyn box
app_mode = st.sidebar.selectbox(
    'App Mode',
    ['About','Image','Video']
)

# Resize Images to fit Container
@st.cache_data()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized

theta=0
#---------------------------------------------------------------------------------------------------
def calculate_distance (point1,point2,ratio=1):
    dis= math.sqrt((point1[0] -point2[0])**2 + (point1[1]- point2[1])**2)
    return int(dis*ratio)
#---------------------------------------------------------------------------------------------------
def between_points(start, end, ratio= 0.5):
    x1,y1=start
    x2,y2=end
    x_diff=x2-x1
    y_diff=y2-y1
    pointx=x1+ x_diff*ratio
    pointy=y1+ y_diff*ratio
    return (int(pointx),int(pointy))
#---------------------------------------------------------------------------------------------------
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle
#---------------------------------------------------------------------------------------------------
def angle_between_lines(start1, end1, start2, end2):
    # Calculate the vectors a and b
    a = (end1[0] - start1[0], end1[1] - start1[1])
    b = (end2[0] - start2[0], end2[1] - start2[1])
    # magnitude_a = math.sqrt(a[0]**2 + a[1]**2)
    # magnitude_b = math.sqrt(b[0]**2 + b[1]**2)
    dot_product = a[0] * b[0] + a[1] * b[1]
    cross_product = a[0] * b[1] - a[1] * b[0]
    angle_degrees = math.degrees(math.atan2(cross_product, dot_product))
    return angle_degrees-90
#-------------------------------------------------------------------------------------------------
def pixel_to_normalized(pixel_coord):
    return mp.solutions.hands.HandLandmark.MPHandLandmark(
        x=pixel_coord[0] / h, # normalized x-coordinate
        y=pixel_coord[1] / w, # normalized y-coordinate
        z=0.0) 
#---------------------------------------------------------------------------------------------------
# Define the rotation matrix
def rot_point(point, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    # ref: https://en.wikipedia.org/wiki/Rotation_matrix
    rotation_matrix = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])

    point_rot = np.dot(rotation_matrix, np.array(point)).astype(int)
    return point_rot

#---------------------------------------------------------------------------------------------------
def get_coordinates_before_rotation(rotation_degree, coordinates_after_rotation, center_of_rotation):
    # convert degree to radians
    theta = np.radians(rotation_degree)

    R_inv = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])

    coordinates_at_origin = coordinates_after_rotation - center_of_rotation
    coordinates_before_rotation = np.dot(R_inv, coordinates_at_origin.T).T
    coordinates_before_rotation += center_of_rotation

    return coordinates_before_rotation
#---------------------------------------------------------------------------------------------------
# def dist_from_ref(p='point', ref_point=ref_point,theta=theta):
#     ref_point = (0,image.shape[0] / 2)
#     # Calculate distances to reference point
#     dist1 = np.linalg.norm([p - ref_point])

#     # Rotate non-reference points around reference pointq
#     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                 [np.sin(theta), np.cos(theta)]])

#     point1_rotated = rotation_matrix.dot(p - ref_point) + ref_point

#     # Calculate distances to rotated non-reference points
#     dist1_rotated = np.linalg.norm(point1_rotated - ref_point)
#     return  dist1_rotated
#---------------------------------------------------------------------------------------------------    
def distance_prependicular (start,end,given_point,distance):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    dist = np.sqrt(dx*dx + dy*dy)
    dx /= dist
    dy /= dist
    pointx = (distance)*dy
    pointy = (distance)*dx

    pointx = given_point[0] + pointx
    pointy= given_point[1] -  pointy

    return (round(pointx), round(pointy))

#---------------------------------------------------------------------------------------------------
# put text and draw circle for our acupuncture points
def txt_circle(point='name of point',txt='name of point',color=0):
    colors={
        0:(0,255,0),
        1:(100,250,40),
        2:(0,0,255),
        3:(100,240,255),
        4:(255,255,0),
        5:(0,255,255),
        6:(114, 219, 57),
        7:(200, 184, 30),
        8:(230 ,40,250),
        9:(255,0,255),  
    }
    cv2.circle(frame, (point[0],point[1]), 1, colors[color], -1)
    # cv2.putText(frame, '.' ,(point[0],point[1]) , cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0), 1, cv2.LINE_AA)
    cv2.putText(frame, txt ,(point[0]-13,point[1]+7) , cv2.FONT_HERSHEY_PLAIN, 0.4,(60,60,60), 0, cv2.LINE_AA)
#---------------------------------------------------------------------------------------------------
def get_points(n, landmark="pose_landmarks"):
    if landmark == "hand_landmarks":
        landmarks = hand_landmarks.landmark
    elif landmark == "face_landmarks":
        landmarks = face_landmarks.landmark
    else:
        landmarks=results_pose.pose_landmarks.landmark
        
    points = []
    for i in range(n):
        point = [landmarks[i].x, landmarks[i].y]
        point = np.multiply(point, [w, h]).astype(int)
        points.append(point)
    return np.array(points)
#---------------------------------------------------------------------------------------------------
def get_hand_type(results, hand_index):
    handedness = results.multi_handedness[hand_index]
    if handedness.classification[0].label == "Right":
        return 'Right'
    elif handedness.classification[0].label == "Left":
        return 'Left'
    else:
        return None
#---------------------------------------------------------------------------------------------------
def angle_between(v1, v2):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle

def determine_direction(hand=True, face=False , handtype='left'):
    if hand:
        # Get the landmarks of the palm and the tips of the index and pinky fingers
        palm_landmark = hand_landmarks.landmark[0]
        index_finger_tip_landmark = hand_landmarks.landmark[8]
        pinky_tip_landmark = hand_landmarks.landmark[20]
        thumb_finger_tip_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        # Calculate the vectors between the palm and the finger tips
        index_vector = np.array([index_finger_tip_landmark.x - palm_landmark.x,
                                index_finger_tip_landmark.y - palm_landmark.y,
                                index_finger_tip_landmark.z - palm_landmark.z])
        
        pinky_vector = np.array([pinky_tip_landmark.x - palm_landmark.x,
                                pinky_tip_landmark.y - palm_landmark.y,
                                pinky_tip_landmark.z - palm_landmark.z])
        
        thumb_vector = np.array([thumb_finger_tip_landmark.x - palm_landmark.x,
                                thumb_finger_tip_landmark.y - palm_landmark.y,
                                thumb_finger_tip_landmark.z - palm_landmark.z])
        if handtype=='left':
            index_vector = np.array([(1-index_finger_tip_landmark.x) - (1-palm_landmark.x),
                                    index_finger_tip_landmark.y - palm_landmark.y,
                                    index_finger_tip_landmark.z - palm_landmark.z])
            
            pinky_vector = np.array([(1-pinky_tip_landmark.x) - (1-palm_landmark.x),
                                    pinky_tip_landmark.y - palm_landmark.y,
                                    pinky_tip_landmark.z - palm_landmark.z])
            
            thumb_vector = np.array([(1-thumb_finger_tip_landmark.x) - (1-palm_landmark.x),
                                    thumb_finger_tip_landmark.y - palm_landmark.y,
                                    0])
            
        normal_vector = np.cross(index_vector, pinky_vector)
        angle = angle_between(normal_vector, [0, 0, 1])
        angle2= angle_between(normal_vector, [1, 0, 0])

        if angle < np.pi/2 + np.pi/5:
            return ["Front",math.degrees(angle),math.degrees(angle2)]
        else:
            return ["Back",math.degrees(angle),math.degrees(angle2)]
    else:
        # Get the landmarks of the palm and the tips of the index and middle fingers
        nose_landmark = face_landmarks.landmark[2]
        left_eye_tip_landmark = face_landmarks.landmark[133]
        right_eye_tip_landmark = face_landmarks.landmark[463]
        # thumb_finger_tip_landmark = face_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        # Calculate the vectors between the nose and the finger tips
        lefteye_vector = np.array([left_eye_tip_landmark.x - nose_landmark.x,
                                left_eye_tip_landmark.y - nose_landmark.y,
                                left_eye_tip_landmark.z - nose_landmark.z])
        
        righteye_vector = np.array([right_eye_tip_landmark.x - nose_landmark.x,
                                right_eye_tip_landmark.y - nose_landmark.y,
                                right_eye_tip_landmark.z - nose_landmark.z])
        
        # Calculate the normal vector of the palm surface
        normal_vector = np.cross(lefteye_vector, righteye_vector)

        # Calculate the angle between the normal vector and the y-axis, in radians
        angle_radians = angle_between(normal_vector, [1, 0, 0])
        angle =math.degrees(angle_radians)
        if angle<75:
            return "Face is facing left"
        if angle>120:
            return "Face is facing right"
        else:
            return "Face is facing center"
# About Page

if app_mode == 'About':
    st.markdown('''
                ## Face Mesh \n
                In this application we are using **MediaPipe** for showing acupoints on face. **StreamLit** is to create 
                the Web Graphical User Interface (GUI) \n
                
                - [Github](https://github.com/mpolinowski/streamLit-cv-mediapipe) \n
    ''')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Image Page

elif app_mode == 'Image':
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Detected Faces**")
    kpil_text = st.markdown('0')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')

    ## Output
    st.markdown('## Output')
    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count=0

    ## Dashboard
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, #Set of unrelated images
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence
    ) as face_mesh:

            results = face_mesh.process(image)
            out_image=image.copy()

            #Face Landmark Drawing
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1

                mp.solutions.drawing_utils.draw_landmarks(
                    image=out_image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec
                )

                kpil_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)

            st.subheader('Output Image')
            st.image(out_image, use_column_width=True)

# Video Page

elif app_mode == 'Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox('Recording', True)

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')

    ## Get Video
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            video = cv.VideoCapture(0)
        else:
            video = cv.VideoCapture(DEMO_VIDEO)
            temp_file.name = DEMO_VIDEO

    else:
        temp_file.write(video_file_buffer.read())
        video = cv.VideoCapture(temp_file.name)

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv.CAP_PROP_FPS))

    ## Recording
    codec = cv.VideoWriter_fourcc('a','v','c','1')
    out = cv.VideoWriter('output1.mp4', codec, fps_input, (width,height))

    st.sidebar.text('Input Video')
    st.sidebar.video(temp_file.name)

    fps = 0
    i = 0

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

    kpil, kpil2, kpil3 = st.columns(3)

    with kpil:
        st.markdown('**Frame Rate**')
        kpil_text = st.markdown('0')

    with kpil2:
        st.markdown('**Detected Faces**')
        kpil2_text = st.markdown('0')

    with kpil3:
        st.markdown('**Image Resolution**')
        kpil3_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)
    ## Face Mesh
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_faces,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence) as face_mesh:

        prevTime = 0

        while video.isOpened():
            i +=1
            ret, frame = video.read()
            if not ret:
                continue
            h, w, c = frame.shape
            fps = video.get(cv2.CAP_PROP_FPS)
            
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            # results = hands.process(frame)
            # results_pose=pose.process(image)

            results_face = face_mesh.process(frame)
            
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results_face.multi_face_landmarks:
                for i, face_landmarks in enumerate(results_face.multi_face_landmarks):
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        # cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.05, (0, 250, 50), 0, cv2.LINE_AA,False)
                        face=determine_direction(hand=False,face=True)
                        cv2.putText(frame, str(face), (int(h/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 0, cv2.LINE_AA,False)

                    points_face = get_points(468,landmark= 'face_landmarks')
                    theta_face= angle_between_lines(points_face [9],points_face [0],(640,240),(0,240))
                    B_cun= calculate_distance(points_face [54],points_face [284],1/9)

                    GB1= [[points_face [446][0],points_face [359][1]],[points_face [226][0],points_face [130][1]]]
                    GB2= [points_face [323],  points_face [93]]
                    GB14= [points_face [299],  points_face [69]]

                    CV24 = points_face [200]

                    LI19 = [between_points(points_face [393],points_face [167],1.2),between_points(points_face [167],points_face [393],1.2)]
                    LI20 = [points_face [358],points_face [129]]

                    GV25 =  points_face [4]
                    GV27 =  points_face [0]
                    GV26 =  between_points(LI19[0],LI19[1])

                    ST1=[points_face [450],points_face [230]]
                    ST2=[points_face [330],points_face [101]]
                    ST4= [points_face [287], points_face [57]]
                    
                    ST3=[points_face [426], points_face [206]]
                    ST5= [between_points(points_face [365],points_face [367]),between_points(points_face [136],points_face [138])]

                    ST6= [points_face [288],points_face [58]]  
                    ST7= [between_points(points_face [366],points_face [454],1/3),between_points(points_face [137],points_face [234],1/3)]  

                    SI18= between_points(points_face [187], points_face [205],0.33),between_points(points_face [411], points_face [425],0.33)
                
                    BL1=[points_face [413],points_face [189]]
                    BL2=[points_face [285],points_face [55]]

                    TE23=[points_face [300],points_face [70]]

                    if face == 'Face is facing center':
                        cv2.putText(frame, f"Angle:{theta_face:.1f} deg", (int(h/2)+25, 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)

                        # 0:right side 1= left side
                        txt_circle(CV24,'CV24',9)
                        txt_circle(GV25,'GV25',7)
                        txt_circle(GV26,'GV26',7)
                        txt_circle(GV27,'Gv27',7)

                        txt_circle(LI19 [0],'LI19')
                        txt_circle(ST1  [0],'ST1',1)
                        txt_circle(ST2  [0],'ST2',1)
                        txt_circle(ST3  [0],'ST3',1)
                        txt_circle(ST4  [0],'ST4',1)
                        txt_circle(BL1  [0],'BL1',5)
                        txt_circle(BL2  [0],'BL2',5)
                        txt_circle(SI18 [0],'SI18',3)
                        txt_circle(GB1  [0],'GB1',6)
                        txt_circle(GB14 [0],'GB14',6)

                        txt_circle(ST1  [1],'ST1',1)
                        txt_circle(ST2  [1],'ST2',1)
                        txt_circle(ST3  [1],'ST3',1)
                        txt_circle(BL1  [1],'BL1',5)
                        txt_circle(BL2  [1],'BL2',5)
                        txt_circle(ST4  [1],'ST4',1)
                        txt_circle(GB14 [1],'GB14',6)
                        txt_circle(GB1  [1],'GB1',6)
                        txt_circle(LI19 [1],'LI19')
                        txt_circle(SI18 [1],'SI18',3)
                        
                    elif face == 'Face is facing right':
                        txt_circle(LI20  [1],'LI20')
                        txt_circle(TE23  [1],'TE23',2)
                        txt_circle(ST5   [1],'ST5',1)
                        txt_circle(ST6   [1],'ST6',1)
                        txt_circle(ST7   [1],'ST7',1)
                        txt_circle(GB2   [1],'GB2',6)

                    else:
                        txt_circle(LI20 [0],'LI20')
                        txt_circle(TE23 [0],'TE23',2)
                        txt_circle(ST5  [0],'ST5',1)
                        txt_circle(ST6  [0],'ST6',1)
                        txt_circle(ST7  [0],'ST7',1)
                        txt_circle(GB2  [0],'GB2',6)

                        face_count += 1

                        # mp.solutions.drawing_utils.draw_landmarks(
                        #     image=frame,
                        #     landmark_list=face_landmarks,
                        #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        #     landmark_drawing_spec=drawing_spec,
                        #     connection_drawing_spec=drawing_spec
                        # )

                # FPS Counter
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(frame)

                # Dashboard
                kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
                                 unsafe_allow_html=True)

                frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame,channels='BGR', use_column_width=True)

