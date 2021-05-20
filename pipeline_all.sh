dialogue_ver='t5'
tts_ver='mellotron'
talking_head_ver='makeittalk'

read input_question
#nlp dialog
python3 dialog.py --mode 'nlp_dialog' --ver $dialogue_ver --input_txt $input_question
#text-to-speech
python3 dialog.py --mode 'tts' --ver $tts_ver
#generate talkinghead
python3 dialog.py --mode 'talking head' --ver talking_head_ver
