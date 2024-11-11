
col_len = 109
confidence = 0.6
bodyparts = {
            'proboscis1':(1,2,3), 'proboscis2':(4,5,6), 'proboscis3' : (7,8,9),
            'head': (10,11,12),
            'antennaR':(13,14,15) , 'antennaL' :(16,17,18),
            'thorax' : (19,20,21),
            'abdomenC':(22,23,24), 'abdomenR' : (25,26,27),'abdomenL' : (28,29,30),
            'bottom' : (31,32,33),
            'rightForeLeg1' : (34,35,36),'rightForeLeg2' : (37,38,39),'rightForeLeg3' : (40,41,42), 'rightForeLeg4':(43,44,45),
            'leftForeLeg1' : (46,47,48),'leftForeLeg2' : (49,50,51),'leftForeLeg3' : (52,53,54), 'leftForeLeg4' : (55,56,57) ,
            'rightMidleg1' : (58,59,60),'rightMidleg2' : (61,62,63),'rightMidleg3' : (64,65,66), 'rightMidleg4' : (67,68,69),
            'leftMidleg1' : (70,71,72),'leftMidleg2' : (73,74,75),'leftMidleg3' : (76,77,78), 'leftMidleg4' : (79,80,81),
            'rightHindleg1' : (82,83,84),'rightHindleg2' : (85,86,87),'rightHindleg3' : (88,89,90), 'rightHindleg4' : (91,92,93),
            'leftHindleg1' : (94,95,96),'leftHindleg2' : (97,98,99),'leftHindleg3' : (100,101,102), 'leftHindleg4' : (103,104,105),
            'stylet' : (106,107,108)
            }

filtered_bodyparts = {
                        'proboscis1' : (0,1) , 'head' : (2,3), 'abdomenR' : (4,5) , 'abdomenL' : (6,7) , 'bottom' : (8,9) , 'rightForeLeg1' : (10,11),'rightForeLeg3':(12,13),
                        'leftForeLeg1' : (14,15) ,'leftForeLeg3':(16,17), 'rightMidleg1' : (18,19),'rightMidleg3':(20,21), 'leftMidleg1' : (22,23) ,'leftMidleg3':(24,25), 'rightHindleg1' : (26,27) , 'leftHindleg1' : (28,29)
                     }

exclude = [
            'proboscis3','thorax','antennaR','proboscis2','rightHindleg2', 
            'leftHindleg2' ,'antennaL','abdomenC','rightForeLeg4','leftForeLeg4',
            'rightMidleg4','leftMidleg4','rightHindleg3','rightHindleg4', 'leftHindleg3','leftHindleg4','stylet', 'abdomenR', 'abdomenL',
            'rightForeLeg2','leftForeLeg2',  'rightMidleg3','leftMidleg3','rightMidleg2','leftMidleg2', 
            'leftHindleg1','rightHindleg1'
          ]
