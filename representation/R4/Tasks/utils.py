import os
try: import _pickle as pickle 
except ImportError: import pickle
import json
import itertools
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

midi2key = None
key2midi = None
finger2key = None
key2finger = None

def create_midi_dictionary():
    midi_notes = [x for x in range(21,108+1)]
    comb_notes = []

    for i in midi_notes:
        comb_notes.append([i])
        for j in range(i+1,midi_notes[-1]+1):
            if(abs(i-j)>20): break
            comb_notes.append([i,j])

            for k in range(j+1,midi_notes[-1]+1):
                if(abs(j-k)>20): break
                comb_notes.append([i,j,k])

                for l in range(k+1,midi_notes[-1]+1):
                    if(abs(k-l)>20): break
                    comb_notes.append([i,j,k,l])

                    for m in range(l+1,midi_notes[-1]+1):
                        if(abs(l-m)>20): break
                        comb_notes.append([i,j,k,l,m])

    key2midi = {k:v for k,v in enumerate(comb_notes)}
    with open(os.path.join(__location__,"midiDictionary.json"), 'w') as f:
        key2midi= json.dump(key2midi,f)
        print('dictionary saved successfully to file')

def create_finger_dict():
    finger = [0,1,2,3,4]
    comb_finger = [tuple([i]) for i in finger]
    for i in range(2,6):
        comb_finger.extend(itertools.combinations(finger,i))
    finger2key = {v:k for k,v in enumerate(comb_finger)}
    key2finger = {v:k for k,v in finger2key.items()}
    return finger2key, key2finger

def load_json_hook(x):
    return {int(k,10):tuple(v) for k, v in x.items()}

finger2key, key2finger = create_finger_dict()

music_keys = ["C_major","Db_major","D_major","Eb_major","E_major","F_major","F#_major","G_major","Ab_major", "A_major","Bb_major","B_major",
              "C_minor","C#_minor","D_minor","D#_minor","E_minor","F_minor","F#_minor","G_minor","G#_minor","A_minor","Bb_minor","B_minor"]
key2sign = {k:v for k,v in enumerate(music_keys)}
sign2key = {v:k for k,v in enumerate(music_keys)}