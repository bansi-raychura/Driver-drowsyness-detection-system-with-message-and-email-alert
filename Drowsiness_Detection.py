from scipy.spatial import distance	 		#imported distance function from scipy.spatial library
from imutils import face_utils 				#imported face_utils function from imutils library
import imutils								#imported the imputils
import dlib 								#imprted dlib library
import cv2									#import cv2
from time import sleep
import Rpi.GPIO as GPIO
import smtplib
import requests
import json
from urllib.request import urlopen
from geopy.geocoders import Nomination
import traceback
import sys
def eye_aspect_ratio(eye):
	dis1 = distance.euclidean(eye[1], eye[5]) #get the distance between eyes two points"""
	dis2= distance.euclidean(eye[2], eye[4])  #same 
	dis3 = distance.euclidean(eye[0], eye[3]) #saME
	ear = (dis1 + dist2) / (2.0 * dis3)		  #gives the eye aspect ration from equeution
	return ear
url ="https://www.fast2sms.com/dev/bulk"
thresh = 0.25								#threshold value for ear comparition
frame_check = 8							#number of frame which is threshold to give alert
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(26,GPIO.OUT)
detect = dlib.get_frontal_face_detector()   #it asign the detector fuction
predict = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
smtpuser="<sender_email_id>"
password="<sender_password>"
add="reciver_email_id"
fromadd=smtpuser
subject="alert"
header='To:'+add+'\n'+'From: '+fromadd +'\n'+'subject:' +subject
body='hiii I am deiver Drowsyness detection,<user_name> is sriving now and and he is drowsy please call him/her or contact them'

my_data = {
     # Your default Sender ID
    'sender_id': 'FSTSMS', 
    
     # Put your message here!
    'message': 'This is a test message', 
    
    'language': 'english',
    'route': 'p',
    
    # You can send sms to multiple numbers
    # separated by comma.
    'numbers': '9999999999, 7777777777, 6666666666'    
}
headers = {
    'authorization': 'YOUR API KEY HERE',			#enter your api key here
    'Content-Type': "application/x-www-form-urlencoded",
    'Cache-Control': "no-cache"
}
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]#savs the left eyes lanmarks(position)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]#savs the right eyes lanmarks(position)
cap=cv2.VideoCapture(0)						#captures the live stream video									
#cap=cv2.VideoCapture("first.mp4")			#capture the static camera
flag=0										#flag for counting
f_num=0
while True:
	ret, frame=cap.read()					#it returns the ret(true,false) and frame
	if f_num%3==0:
		frame = imutils.resize(frame, width=450)#it resize the frame 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#it converts frmae's bgr color to gray color
		subjects = detect(gray, 0)				#it detects face from the gray image
		for subject in subjects:
			shape = predict(gray, subject)		#it predicts the shape from the subject 
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array
			leftEye = shape[lStart:lEnd]		#it gives the shape co-ordinates of left-eyes	
			rightEye = shape[rStart:rEnd]		
			leftEAR = eye_aspect_ratio(leftEye) #it saves the ear value for lefteye
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0	#it takes the average of ear value
			leftEyeHull = cv2.convexHull(leftEye)#it creates the convexhull shape for lefteye
			rightEyeHull = cv2.convexHull(rightEye)#it creates the convexhull shape for righteye
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)#it draws the left eye convexhull on frame
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)#it draws the left eye convexhull on frame
			if ear < thresh:					#it compairs the ear and threshlod ratio
				flag += 1
				print (flag)
				if flag >= frame_check:			#it compairs the flag and frame check 
					cv2.putText(frame, "****************ALERT!****************", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#put text on the frame
					cv2.putText(frame, "****************ALERT!****************", (10,325),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					#print ("Drowsy")
					GPIO.output(26,GPIO.HIGH)
					if(flag==frame_check):
						s=smtplib.SMTP('smtp.gmail.com',587)
						s.ehlo()
						s.starttls()
						s.echo()
						s.login(smtpuser,password)
						s.sendmail(fromadd,add,header+ '\n'+body)
						s.quit()
						try:
							response = requests.request("POST",
                            url,
                            data = my_data,
                            headers = headers)
#
							#load json data from source
							returned_msg = json.loads(response.text)
							
							# print the send message
							print(returned_msg['message'])
						except:
							exc_type,exc_value,exc_traceback=sys.exc_info()
							s=smtplib.SMTP('smtp.gmail.com',587)
							s.ehlo()
							s.starttls()
							s.ehlo()
							s.login(smtpuser,password)
							s.sendmail(fromadd,add,header+'\n'+str(exc_value))
							s.quit()
					else:
						GPIO.output(26,GPIO.LOW)

			else:
				flag = 0
		cv2.imshow("Frame", frame)				#display the frame
		key = cv2.waitKey(1) & 0xFF				#wait to enter key tostop execution
		if key == ord("q"):
			break
	f_num+=1
cv2.destroyAllWindows()						#distroys all windows											
cap.stop()									#stops the camera			
