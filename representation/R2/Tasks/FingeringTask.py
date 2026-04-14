import os
import utils
import numpy as np
import music21
import itertools
from decoder_operator import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def defaultRules(notes, fingering, hand):
    difc = 0.0
    isFirst = True
    #print(notes)
    notes = [[-1]] + notes
    fingering = [[-1]] + fingering

    for i in range(len(notes)-2):
        #prev_note, prev_finger, curr_note, curr_finger, next_note, peek_note = self.state
        if(i+3 == len(notes)):
            peek_note = [-1]
        else:
            peek_note = notes[i+3]
        prev_note, curr_note, next_note = notes[i:i+3]
        prev_finger, curr_finger, action_finger = fingering[i:i+3]
        #print(prev_note, prev_finger, curr_note, curr_finger, next_note, action_finger, peek_note)
        assert len(curr_note) == len(curr_finger) , f"{len(curr_note)},{len(curr_finger)}"
        assert len(next_note) == len(action_finger), f"{len(next_note) },{len(action_finger)}"

        isPrevMono = True
        isCurrNextMono = True 
        isActionMono = True

        if(len(prev_note) > 1):
            isPrevMono = False 
        if(len(curr_note) > 1 or len(next_note) > 1) : isCurrNextMono = False
        if(len(next_note)>1): isActionMono = False

        curr_next_combination = itertools.product(zip(curr_note,curr_finger),zip(next_note,action_finger))
        reward = 0.0


        # ----------- Mono only Case
        if(isCurrNextMono == True):
            cNote = FingeringTask.note_alter(curr_note[0])
            nNote  = FingeringTask.note_alter(next_note[0])
            cFinger = curr_finger[0]
            aFinger = action_finger[0]
            interval = nNote - cNote
            minRel, minComf, minPrac, maxRel, maxComf, maxPrac = FingeringTask.get_distance(cFinger, aFinger,hand)

            if(isFirst == False and isPrevMono == True):
                pNote = FingeringTask.note_alter(prev_note[0])
                pFinger = prev_finger[0]
                pn_interval = nNote - pNote
                pn_minRel, pn_minComf, pn_minPrac, pn_maxRel, pn_maxComf, pn_maxPrac = FingeringTask.get_distance(pFinger, aFinger,hand)

                # Rule 3
                temp_reward = 0.0
                if (pn_interval < pn_minComf or pn_interval > pn_maxComf): # Full change of hand
                    temp_reward -= 1
                    if(pNote < cNote  and cNote < nNote 
                       and cFinger == 0 and (pn_interval < pn_minPrac or pn_interval > pn_maxPrac)): 
                        temp_reward -= 1

                    if(pNote == nNote and pFinger != aFinger):
                        temp_reward -= 1

                assert temp_reward <= 0, "rule 3"
                reward += temp_reward
                #print("rule 3 : ", temp_reward)

                # Rule 4
                temp_reward = 0.0
                if(pn_interval < pn_minComf):
                    temp_reward -= (pn_minComf - pn_interval) 
                elif(pn_interval > pn_maxComf):
                    temp_reward -= (pn_interval - pn_maxComf)
                assert temp_reward <= 0, "rule 4"
                reward += temp_reward  
                #print("rule 4 : ", temp_reward)

                # Rule 12
                temp_reward = 0.0
                #print(pNote, nNote, pFinger, aFinger, cNote)
                if(pNote != nNote and pFinger == aFinger and pNote < cNote and cNote < nNote):
                    temp_reward -= 1
                assert temp_reward <= 0, "rule 12"
                reward += temp_reward 
                #print("rule 12 : ", temp_reward)

            # Rule 6
            temp_reward = 0.0
            if((cFinger,aFinger) in [(2,3),(3,2)]):
                temp_reward -= 1
            assert temp_reward <= 0, "rule 6"
            reward += temp_reward  
            #print("rule 6 : ", temp_reward)

            # Rule 7
            temp_reward = 0.0
            if((cNote%2 == 0 and cFinger== 2 and nNote%2 ==1 and aFinger == 3) or 
               (cNote%2 == 1 and cFinger== 3 and nNote%2 ==0 and aFinger == 2)):
                temp_reward += -1
            assert temp_reward <= 0, "rule 7"
            reward += temp_reward  
            #print("rule 7 : ", temp_reward)

            if(peek_note != [-1] and len(peek_note) == 1):
                peekNote = FingeringTask.note_alter(peek_note[0])

                # Rule 8
                temp_reward = 0.0
                if(nNote%2 == 1 and aFinger == 0):
                    temp_reward -= 0.5
                    if(cNote%2 ==  0): temp_reward -= 1
                    if(peekNote%2 ==  0): temp_reward -= 1
                assert temp_reward <= 0, "rule 8"
                reward += temp_reward
                #print("rule 8 : ", temp_reward)

                # Rule 9
                temp_reward = 0.0
                if(nNote%2 == 1 and aFinger == 4):
                    if(cNote%2 ==  0): temp_reward -= 1
                    if(peekNote%2 ==  0): temp_reward -= 1
                assert temp_reward <= 0, "rule 9"
                reward += temp_reward
                #print("rule 9 : ", temp_reward)
            
            """
            nNote%2 = 0 white key
            nNote%2 = 1 black key
            """
            # Rule 10
            temp_reward = 0.0
            if((cNote%2 == 1 and cFinger != 0 and nNote%2 == 1 and aFinger == 0) or
              (cNote%2 == 0 and cFinger != 0 and nNote%2 == 0 and aFinger == 0)):
                temp_reward -= 1
            # Rule 11
            if(cNote%2 == 0 and cFinger != 0 and nNote%2 == 1 and aFinger == 0):
                temp_reward -= 2
            assert temp_reward <= 0, "rule 10 and 11"
            reward += temp_reward
            #print("rule 10 and 11 : ", temp_reward)


        # Rule 5
        temp_reward = 0.0
        if(isActionMono == True):
            if(action_finger[0] == 3):
                temp_reward -= 1
        assert temp_reward <= 0, "rule 5"
        reward += temp_reward  
        #print("rule 5 : ", temp_reward)
        # --------- End of Mono only case


        # Rule 15
        temp_reward = 0.0
        if(list(curr_note) == list(next_note) and list(curr_finger) != list(action_finger)):
            temp_reward -= 1
        #print("rule 15 : ", temp_reward)


        # --------- All Case Current and Next Distance Related
        for [cNote,cFinger],[nNote, aFinger] in curr_next_combination:       
            cNote = FingeringTask.note_alter(cNote)
            nNote  = FingeringTask.note_alter(nNote)
            interval = nNote - cNote
            minRel, minComf, minPrac, maxRel, maxComf, maxPrac = FingeringTask.get_distance(cFinger, aFinger,hand)

            # Rule 1 - 
            temp_reward = 0.0
            if(interval > maxComf):
                temp_reward -= (2 * (interval-maxComf))  # more than max comfortable
            elif(interval < minComf):
                temp_reward -= (2 * (minComf-interval))  # less than min comfortable
            assert temp_reward <= 0, "rule 1"
            reward += temp_reward
            #print("rule 1 : ", temp_reward)

            # Rule 2
            temp_reward = 0.0
            if(interval < minRel):   
                temp_reward -= (minRel - interval)
            elif(interval > maxRel):
                temp_reward -= (interval - maxRel)  
            assert temp_reward <= 0, "rule 2"
            reward += temp_reward
            #print("rule 2 : ", temp_reward)

            # Rule 13
            temp_reward = 0.0
            if(interval < minPrac):   
                temp_reward -= 10*(minPrac - interval)
            elif(interval > maxPrac):
                temp_reward -= 10*(interval - maxPrac)  
            assert temp_reward <= 0, "rule 13"
            reward += temp_reward
            #print("rule 13 : ", temp_reward)


        # --------- All Case Current and Next Distance Related


        # --------- Chord Action Case -- Rule 14
        if(isActionMono == False):
            for [note1,finger1],[note2, finger2] in itertools.combinations(zip(next_note,action_finger),2):
                curr_note_alter = FingeringTask.note_alter(note1)
                next_note_alter = FingeringTask.note_alter(note2)
                interval = next_note_alter - curr_note_alter

                minRel, minComf, minPrac, maxRel, maxComf, maxPrac = FingeringTask.get_distance(finger1,finger2,hand)

                # Apply rule 1,2, 13 with Doubled
                # Rule 1 - 
                temp_reward = 0.0
                if(interval > maxComf):
                    temp_reward -= (2 * (interval-maxComf))  # more than max comfortable
                elif(interval < minComf):
                    temp_reward -= (2 * (minComf-interval))  # less than min comfortable
                assert temp_reward <= 0, "rule 14-r1"
                reward = reward + 2*temp_reward
                #print("rule 14-r1 : ", temp_reward)

                # Rule 2
                temp_reward = 0.0
                if(interval < minRel):   
                    temp_reward -= (minRel - interval)
                elif(interval > maxRel):
                    temp_reward -= (interval - maxRel)  
                assert temp_reward <= 0, "rule 14-r2"
                reward = reward + 2*temp_reward
                #print("rule 14-r2 : ", temp_reward)

                # Rule 13
                temp_reward = 0.0
                if(interval < minPrac):   
                    temp_reward -= 10*(minPrac - interval)
                elif(interval > maxPrac):
                    temp_reward -= 10*(interval - maxPrac)  
                assert temp_reward <= 0, "rule 14-r13"
                reward = reward + 2*temp_reward
                #print("rule 14-r13 : ", temp_reward)
        
        if isFirst == True:
            isFirst = False
        difc += reward
    return (-1)*difc


class MusicSheet(object):
    """
        Loading the PIG dataset format txt
    """
    def __init__(self, file):
        self.filePath=file
        self.title = ""
        self.LHand  = []
        self.RHand = []
        self.L_nNotes = []
        self.R_nNotes = []
        self.music_stream = None

        if(file.split(".")[-1] == "txt"): 
            self.load_file_txt()
        else:
            self.load_file_xml()

        self.dim = [len(self.R_nNotes), len(self.L_nNotes)]


    def load_file_txt(self): 
        count = 0
        with open(os.path.join(__location__,self.filePath), "r") as f:
            f.readline()
            notes_in_same_time = []
            prev_time = '-1'
            prev_hand = '-1'
            for line in f:
                line_splitted = line.split()
                if prev_time == '-1' or (prev_time == line_splitted[1] and prev_hand == line_splitted[-2]):
                    notes_in_same_time.append(music21.pitch.Pitch(line_splitted[3]).midi)#.nameWithOctave))
                else:
                    if(prev_hand == "0"): 
                        # since "0" is right hand, so we store to the right hand 
                        self.RHand.append(notes_in_same_time)
                        self.R_nNotes.append(len(notes_in_same_time))
                        prev_hand = "1"
                    else:
                        self.LHand.append(notes_in_same_time)
                        self.L_nNotes.append(len(notes_in_same_time))
                        prev_hand = "0"
                    notes_in_same_time = []
                    notes_in_same_time.append(music21.pitch.Pitch(line_splitted[3]).midi)#.nameWithOctave))
                prev_time = line_splitted[1]
                prev_hand = line_splitted[-2]
                count+=1
        if(prev_hand == "0"): 
            # since "0" is right hand, so we store to the right hand 
            self.RHand.append(notes_in_same_time)
            self.R_nNotes.append(len(notes_in_same_time))
        else:
            self.LHand.append(notes_in_same_time)
            self.L_nNotes.append(len(notes_in_same_time))
        assert count == sum(self.L_nNotes)+sum(self.R_nNotes), f"Len : {count, sum(self.L_nNotes)+sum(self.R_nNotes)}"
        self.title = self.filePath.split("-")[0][-3:]

    def save_file_txt(self,fingering, ftype = "encoded", **kwargs):
        # From input Txt output Txt
        if(ftype == "encoded"):
            fingering = [list(utils.key2finger[int(i)]) for i in fingering]

        F_Rhand = fingering[:self.dim[0]]
        F_Rhand = [int(x+1) for fin in F_Rhand for x in fin]
        F_Lhand = fingering[self.dim[0]:]
        F_Lhand = [int(-(x+1)) for fin in F_Lhand for x in fin[::-1]]
        assert len(fingering) == sum(self.dim), "finger given is not same as note"
        
        f_counter = [0,0]
        text="// Fingering \n"
        
        with open(os.path.join(__location__,self.filePath), "r") as f:
            f.readline()
            for line in f:
                line_splitted = line.split()
                if line_splitted[-2] == "0":
                    temp = f"{line_splitted[0]}\t{line_splitted[1]}\t{line_splitted[2]}\t{line_splitted[3]}\t{line_splitted[4]}\t{line_splitted[5]}\t{line_splitted[6]}\t{F_Rhand[f_counter[0]]}\n"
                    f_counter[0] +=1
                else:
                    temp = f"{line_splitted[0]}\t{line_splitted[1]}\t{line_splitted[2]}\t{line_splitted[3]}\t{line_splitted[4]}\t{line_splitted[5]}\t{line_splitted[6]}\t{F_Lhand[f_counter[1]]}\n"
                    f_counter[1] +=1
                text += temp
        try:
            with open(os.path.join(kwargs["path"]), "w") as f:
                f.write(text)
        except:
            Exception("saving path hasn't been specified yet")
    
    def load_file_xml(self):
        # load from musicxml, xml, mxl file
        xml_score = music21.converter.parse(os.path.join(__location__,self.filePath))
        for note in xml_score.parts[0].flatten().notes:
            
            if(note.tie != None):
                if(note.tie.type == "stop"):
                    continue
                    
            if(note.isNote == False):
                temp = [x.pitch.midi for x in note]    
                self.RHand.append(temp)
                self.R_nNotes.append(len(temp))
            else:
                self.RHand.append([note.pitch.midi])
                self.R_nNotes.append(1)

        for note in xml_score.parts[1].flatten().notes:  
            
            if(note.tie != None):
                if(note.tie.type == "stop"):
                    continue
            if(note.isNote == False):

                temp = [x.pitch.midi for x in note]    
                self.LHand.append(temp)
                self.L_nNotes.append(len(temp))
            else:
                self.LHand.append([note.pitch.midi])
                self.L_nNotes.append(1)
        self.music_stream = xml_score
        self.title = xml_score.metadata.title
        
    def save_file_xml(self, fingering, ftype = "undecoded", **kwargs):
        # input xml stream, output xml stream
        if(ftype == "undecoded"):
            fingering = [list(utils.key2finger[int(i)]) for i in fingering]

        F_Rhand = fingering[:self.dim[0]]
        F_Rhand = [int(x+1) for fin in F_Rhand for x in fin]
        F_Lhand = fingering[self.dim[0]:]
        F_Lhand = [int(x+1) for fin in F_Lhand for x in fin[::-1]]
        assert len(fingering) == sum(self.dim), "finger given is not same as note"

        #self.music_stream.parts[1].flatten().notes[idx].articulations.append(music21.articulations.Fingering(finger))
        f_counter = [0,0]

        for note in self.music_stream.parts[0].flatten().notes:  
            if(note.tie != None):
                if(note.tie.type == "stop"):
                    continue
                    
            if(note.isNote == False):
                for _ in note:
                    note.articulations.append(music21.articulations.Fingering(F_Rhand[f_counter[0]]))
                    f_counter[0] +=1
            else:
                note.articulations.append(music21.articulations.Fingering(F_Rhand[f_counter[0]]))
                f_counter[0] +=1

        for note in self.music_stream.parts[1].flatten().notes:  
            if(note.tie != None):
                if(note.tie.type == "stop"):
                    continue
            if(note.isNote == False):
                for _ in note:
                    note.articulations.append(music21.articulations.Fingering(F_Lhand[f_counter[1]]))
                    f_counter[1] +=1
            else:
                note.articulations.append(music21.articulations.Fingering(F_Lhand[f_counter[1]]))
                f_counter[1] +=1

        #self.music_stream.show("musicxml")
        self.music_score.write("musicxml",kwargs["path"])
    
    
    def save_file(self,fingering,filetype="musicxml",**kwargs):
        if(filetype == "musicxml"): self.save_file_xml(fingering, ftype="undecoded" , **kwargs)
        else: self.save_file_txt(fingering,**kwargs)
        return True

    def get_nNotes(self):
        return np.array(self.R_nNotes + self.L_nNotes)

    def set_fingering(self, fingering):
        music_stream = self.music_stream
        music_stream.getElementsByClass('Note')

    def isStreamAvailable(self):
        return False if self.music_stream == None else True
        
    def view(self, fingering, mode):
        # Note: haven't implemented the append fingering to xml
        if(not self.isStreamAvailable()):
            raise Exception("the initial file is not in musicxml format")
        try:
            self.music_stream.show(mode)
        except:
            self.music_stream.show("text")
            print("Install Lilypond or Musescore to get better viewing in the application")
    
    def generate_ground_truth(self, fingering, ftype="undecoded", **kwargs):
        # Generate from musicxml to txt PIG Dataset style
        if(ftype == "undecoded"):
            fingering = [list(utils.key2finger[int(i)]) for i in fingering]

        F_Rhand = fingering[:self.dim[0]]
        F_Rhand = [int(x+1) for fin in F_Rhand for x in fin]
        F_Lhand = fingering[self.dim[0]:]
        F_Lhand = [int(-(x+1)) for fin in F_Lhand for x in fin[::-1]]
        assert len(fingering) == sum(self.dim), "finger given is not same as note"

        offsetR = []
        offsetL = []
        dataR = []
        dataL = []
        for e in self.music_stream.parts[0].flatten().recurse().notes:
            offsetR.append(e.offset)
            if(len(e.pitches)> 1):
                for n in e.pitches:
                    x = n.nameWithOctave
                    notewithOctave = str(x) #if str(x)[1]!='-' else str(x)[0]+"b"+str(x)[2:]
                    dataR.append([e.offset, e.quarterLength, notewithOctave])
            else:
                x =  e.pitches[0].nameWithOctave
                notewithOctave = str(x) #if str(x)[1]!='-' else str(x)[0]+"b"+str(x)[2:]
                dataR.append([e.offset, e.quarterLength, notewithOctave])
                
        for e in self.music_stream.parts[1].flatten().recurse().notes:
            offsetL.append(e.offset)
            if(len(e.pitches)> 1):
                for n in e.pitches:
                    x = n.nameWithOctave
                    notewithOctave = str(x) #if str(x)[1]!='-' else str(x)[0]+"b"+str(x)[2:]
                    dataL.append([e.offset, e.quarterLength, notewithOctave])
            else:
                x =  e.pitches[0].nameWithOctave
                notewithOctave = str(x) #if str(x)[1]!='-' else str(x)[0]+"b"+str(x)[2:]
                dataL.append([e.offset, e.quarterLength, notewithOctave])

        setOffset = sorted(set(offsetL+offsetR))

        text="// PIT Dataset Ver : 0.1\n"
        i=0
        f_counter = [0,0]

        for j, itr in enumerate(setOffset):
            while itr == dataR[f_counter[0]][0]:
                temp = f"{i}\t{dataR[f_counter[0]][0]}\t{dataR[f_counter[0]][0]+dataR[f_counter[0]][1]}\t{dataR[f_counter[0]][2]}\t{64}\t{80}\t0\t{F_Rhand[f_counter[0]]}\n"
                text += temp
                i+=1
                if(f_counter[0]+1 == len(dataR)): break
                else:
                    f_counter[0] += 1
                
            while itr == dataL[f_counter[1]][0]:
                temp = f"{i}\t{dataL[f_counter[1]][0]}\t{dataL[f_counter[1]][0]+dataL[f_counter[1]][1]}\t{dataL[f_counter[1]][2]}\t{64}\t{80}\t1\t{F_Lhand[f_counter[1]]}\n"
                
                text += temp
                i+=1
                if(f_counter[1]+1 == len(dataL)): break
                else:
                    f_counter[1] += 1
                

        with open(os.path.join(__location__,kwargs['folder'],f"{kwargs['filename']}.txt"), "w") as f:
            f.write(text)
    
    def retrieve_fingering_xml(self):
        fingeringR = []
        fingeringL = []
        for notes in self.music_stream.parts[0].flatten().recurse().notes:
            fingeringR.append(utils.finger2key[tuple([notes.articulations[i].fingerNumber-1 for i in range(len(notes.articulations)) if notes.articulations[i].name =="fingering"])])

        for notes in self.music_stream.parts[1].flatten().recurse().notes:
            fingeringL.append(utils.finger2key[tuple([notes.articulations[i].fingerNumber-1 for i in range(len(notes.articulations)) if notes.articulations[i].name =="fingering"][::-1])])
        return fingeringR + fingeringL

    def retrieve_fingering_txt(self):
        fingeringR = []
        fingeringL = []
        count =0 
        with open(os.path.join(__location__,self.filePath), "r") as f:
            f.readline()
            finger_in_same_time = []
            prev_time = '-1'
            prev_hand = '-1'

            for line in f:
                line_splitted = line.split()
                if prev_time == '-1' or (prev_time == line_splitted[1] and prev_hand == line_splitted[-2]):
                    finger_in_same_time.append(int(line_splitted[-1]))
                else:
                    if(prev_hand == "0"): 
                        # since "0" is right hand, so we store to the right hand 
                        fingeringR.append([x-1 for x in finger_in_same_time])
                        prev_hand = "1"
                    else:
                        fingeringL.append([-(x+1) for x in finger_in_same_time])
                        prev_hand = "0"
                    finger_in_same_time = []
                    finger_in_same_time.append(int(line_splitted[-1]))
                prev_time = line_splitted[1]
                prev_hand = line_splitted[-2]
                count+=1
        if(prev_hand == "0"): 
            # since "0" is right hand, so we store to the right hand 
            fingeringR.append([x-1 for x in finger_in_same_time])
        else:
            fingeringL.append([-(x+1) for x in finger_in_same_time])
        assert count == sum(self.L_nNotes)+sum(self.R_nNotes), f"Len : {count, sum(self.L_nNotes)+sum(self.R_nNotes)}"
        fingeringR_encoded = [utils.finger2key[tuple(sorted(finger))] for finger in fingeringR]
        fingeringL_encoded = [utils.finger2key[tuple(sorted(finger))] for finger in fingeringL]
        self.fingering_encoded = fingeringR_encoded+fingeringL_encoded
        return fingeringR + fingeringL


def lagrangian_penalty(nNotes, nFinDecoded, lambda_=0.4):
    penalty = lambda_ * np.abs(nFinDecoded - nNotes)
    return penalty.sum()

class FingeringTask(object):
    MAX_PRAC = np.array([[0,11,15,16,18],
                    [10,0,7,8,12],
                    [8,-1,0,4,8],
                    [6,-1,-1,0,6],
                    [2,-2,-1,-1,0]
                    ])

    MAX_COMF = np.array([[0,9,13,14,16],
                        [8,0,5,6,10],
                        [6,-1,0,2,6],
                        [4,-1,-1,0,4],
                        [0,-2,-1,-1,0]
                        ])

    MAX_REL = np.array([[0,6,9,11,12],
                        [-1,0,2,4,6],
                        [-3,-1,0,2,4],
                        [-5,-3,-1,0,2],
                        [-7,-5,-3,-1,0]
                        ])

    def __init__(self, file, music_type="repertoire", 
                 decoder_type = "threshold_decoder", use_penalty = False,
                 penalty_type = "lagrangian", rules=defaultRules):
        accepted_file = ["musicxml", "xml", "mxl","txt"]
        # initiate music sheet 
        """
        print(file)
        if(str(file).split(".")[-1] not in accepted_file):
            raise NotImplementedError("the file must be either in musicxml, xml, mxl or txt")
        if(str(file).split(".")[-1] in accepted_file[:-1]):
            print("processing xml file")
        else: print("processing txt file")
        """

        self.musicSheet = MusicSheet(file)
        self.dim = sum(self.musicSheet.dim)
        self.Rdim = self.musicSheet.dim[0]
        self.rules = rules
        self.nNotes = self.musicSheet.get_nNotes()
        self.fingering = self.retrieve_gt()
        self.music_type = music_type 
        self.use_penalty = use_penalty
        self.decoder_type = decoder_type

        if decoder_type == "threshold_decoder":            # PF_baseline
            self.decoder = threshold_decoder
        elif decoder_type == "priority_based_decoder":     # PFB1
            self.decoder = priority_based_decoder
        elif decoder_type == "controlled_softmax_decoder": # PFB2
            self.decoder = controlled_softmax_decoder
        elif decoder_type == "k_block_decoder":            # PFB3
            self.decoder = k_block_decoder
        elif decoder_type == "relaxation_with_threshold":
            self.decoder = relaxation_with_threshold
        else:
            raise ValueError("Decoder is not in the list")

        self.penalty_function = lagrangian_penalty

    def fingering_difficulty(self, fingering, nRawFinger, ftype = "encoded"):
        # decode from key -> fingering and key -> midi
        #fingering = FingeringTask.key2fingerF(fingering)
        if(ftype=="encoded"):   
            fingering = [list(utils.key2finger[int(i)]) for i in fingering]
            for i, finger in enumerate(fingering[:self.Rdim]):
                if self.musicSheet.RHand[i] != sorted(self.musicSheet.RHand[i]):
                    #print("notes", i, self.musicSheet.RHand[i])
                    #print("finger", finger)
                    y = np.argsort(self.musicSheet.RHand[i])
                    yy = np.argsort(y)
                    #print("order", y, yy)
                    fingering[i] = np.array(finger)[yy].tolist()
                    #print(fingering[i])
            
            for i, finger in enumerate(fingering[self.Rdim:]):
                #fingering[self.Rdim+i] = finger.reverse()
                fingering[self.Rdim+i].reverse()
                #print(fingering[self.Rdim+i])
                if self.musicSheet.LHand[i] != sorted(self.musicSheet.LHand[i]):
                    #print("notes", i, self.musicSheet.LHand[i])
                    #print("finger", finger)
                    y = np.argsort(self.musicSheet.LHand[i])[::-1]
                    yy = np.argsort(y)
                    #print("order", y)
                    fingering[self.Rdim+i] = np.array(finger)[yy].tolist()
                    #print(fingering[self.Rdim+i])
        difc_r = 0
        difc_l = 0 
        difc_r = self.rules(self.musicSheet.RHand,fingering[:self.Rdim],"0")
        difc_l = self.rules(self.musicSheet.LHand,fingering[self.Rdim:],"1")
        # print(difc_r,difc_l,self.Rdim,self.dim-self.Rdim)
        #print(difc_r, difc_l)
        difc = difc_r/len(self.musicSheet.R_nNotes) + difc_l/len(self.musicSheet.L_nNotes)
        #print(difc_r, difc_l, difc)
        
        # constraint violation
        if self.use_penalty:
            penalty = self.penalty_function(self.nNotes, nRawFinger)
        else: 
            penalty = 0.0

        return difc, penalty
    
    def decode(self, vec, args=None):
        vec = np.array(vec)
        if(self.decoder_type == "controlled_softmax_decoder"):
            key, nRawFinger = self.decoder(self.nNotes, vec, args)
        else: 
            key, nRawFinger = self.decoder(self.nNotes, vec)
        return key, nRawFinger
    
    def fingering_gt(self, fingering, nRawFinger):
        check = self.musicSheet.fingering_encoded == fingering
        count = len(check)-check.sum()
        penalty = self.penalty_function(self.nNotes, nRawFinger)
        return count, penalty

    def view(self,fingering, mode="musicxml"):
        self.musicSheet.view(fingering, mode)

    def save(self,fingering,filetype="txt",**kwargs):
        self.musicSheet.save_file(fingering,filetype,**kwargs)
    
    def get_dim(self):
        return self.dim
    
    def change_matrix(self, MAX_REL, MAX_COMF, MAX_PRAC):
        return True

    def generate_gt(self,fingering,ftype="undecoded",**kwargs):
        self.musicSheet.generate_ground_truth(fingering,ftype,**kwargs)
        
    def retrieve_gt(self):
        if(self.musicSheet.music_stream == None):
            return self.musicSheet.retrieve_fingering_txt()
        else:
            return self.musicSheet.retrieve_fingering_xml()
         

    @staticmethod
    def get_distance(f,g,hand):
        if(hand == "0"):
            minRel = -1 * FingeringTask.MAX_REL[g][f]
            minComf = -1 * FingeringTask.MAX_COMF[g][f]
            minPrac = -1 * FingeringTask.MAX_PRAC[g][f]

            maxRel = FingeringTask.MAX_REL[f][g]
            maxComf = FingeringTask.MAX_COMF[f][g]
            maxPrac = FingeringTask.MAX_PRAC[f][g]
        else:
            minRel = -1 * FingeringTask.MAX_REL[f][g]
            minComf = -1 * FingeringTask.MAX_COMF[f][g]
            minPrac = -1 * FingeringTask.MAX_PRAC[f][g]

            maxRel = FingeringTask.MAX_REL[g][f]
            maxComf = FingeringTask.MAX_COMF[g][f]
            maxPrac = FingeringTask.MAX_PRAC[g][f]

        return minRel, minComf, minPrac, maxRel, maxComf, maxPrac

    def note_alter(note):
        if (note %12 > 4):
            return note +1 
        return note
        
