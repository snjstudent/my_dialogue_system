dialogue_ver='t5'
tts_ver='mellotron'
talking_head_ver='makeittalk'
gesture_location_ver='Speech_driven_gesture_generation_with_autoencoder'
gesture_fashion_ver='adgan'
gesture_img2img_ver='articulated_animation'

set -e
read input_question

#nlp dialog
python3 dialog.py dialog_nlp -i $input_question $dialogue_ver

#text-to-speech
python3 dialog.py dialog_tts -i $tts_ver

#generate talkinghead
python3 dialog.py dialog_talkinghead -i $talking_head_ver

#gesture location
python3 dialog.py gesture_fromspeech -i $gesture_location_ver

#gesture fashion
python3 dialog.py gesture_location_to_img -i $gesture_fashion_ver

#gesture img2img
python3 dialog.py gesture_img_to_img -i $gesture_img2img_ver

#align face and body
python3 dialog.py align_face_body
