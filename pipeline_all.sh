dialogue_ver='t5'
tts_ver='mellotron'
talking_head_ver='makeittalk'

set -e
read input_question
#nlp dialog
python3 dialog.py dialog_nlp -i $input_question $dialogue_ver
#text-to-speech
python3 dialog.py dialog_tts -i $tts_ver
#generate talkinghead
python3 dialog.py dialog_talkinghead -i $talking_head_ver
