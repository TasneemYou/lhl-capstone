import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(df):
    #Preprocess dataframe wise
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df['Max._Allowable_Loading_Per_Phase']=pd.to_numeric(df['Max._Allowable_Loading_Per_Phase'])
    df.replace({'Capacity\n(KVA)':{'500 (PMU)':'500','25@':'25','500 PMU':'500'}},inplace=True)
    df['Capacity\n(KVA)']=pd.to_numeric(df['Capacity\n(KVA)'])
    df['Max._Allowable_Loading_Per_Phase'].fillna(df['Capacity\n(KVA)']*1000/240/3,inplace=True)
    df['Rated_Power']=df['Max._Allowable_Loading_Per_Phase']*0.8
    df['Final_Conclusion'].fillna('Loading issue',inplace=True)
    df['RED_PHASE_LOADING']=pd.to_numeric(df['RED_PHASE_LOADING'].mask((df['RED_PHASE_LOADING'] == '-') | \
                                                                   (df['RED_PHASE_LOADING'] == 'Nill') | \
                                                                   (df['RED_PHASE_LOADING'] == 'NIL') | \
                                                                   (df['RED_PHASE_LOADING'] == 'FAULT')))
    df['YELLOW_PHASE_LOADING']=pd.to_numeric(df['YELLOW_PHASE_LOADING'].mask((df['YELLOW_PHASE_LOADING'] == '-') |\
                                                                        (df['YELLOW_PHASE_LOADING'] == 'Nill') |\
                                                                        (df['YELLOW_PHASE_LOADING'] == 'NIL') |\
                                                                        (df['YELLOW_PHASE_LOADING'] == 'LINK BLOWN OUT')))
    df['BLUE_PHASE_LOADING']=pd.to_numeric(df['BLUE_PHASE_LOADING'].mask((df['BLUE_PHASE_LOADING'] == '-') |\
                                                                    (df['BLUE_PHASE_LOADING'] == 'Nil')|\
                                                                    (df['BLUE_PHASE_LOADING'] == 'NIL')|\
                                                                    (df['BLUE_PHASE_LOADING'] == 'Nill')))
    df.replace({'CLUSTER':{'C-N':'CN','C-O':'CO','C-K':'CK','C-Q':'CQ','C-J':'CJ','C-B':'CB','C-D':'CD','C-U':'CU','C-S':'CS',\
                       'C-G':'CG','SECTOR 10, BLOCK D, PMT NO. 01,MILLAT TOWN,Korangi Cluste,DTS IN MILLAT TOWN FEEDER':'CK',\
                      'NAME: CHOONA DEPOT D/PMT (LHS),FEDR: C-1 AREA,R-4,GRID: LIAQUTABAD':'CN'}},inplace=True)
    df.replace({'CLUSTER':{'-':np.nan,'0':np.nan}},inplace=True)
    df.replace({'CLUSTER':{0:np.nan}},inplace=True)
    df.replace({'REGION':{'R2-C1':'R2','R2-C2':'R2','R2W':'R2','R2S':'R2','UGM_S-R3':'R3','R-3':'R3','R-4':'R4','R-1':'R1',\
                     'NC':'R1','BSY':'R3'}},inplace=True)
    df.replace({'REGION':{'TLR':'R3'}},inplace=True)
    df.replace({'REGION':{'-':np.nan}},inplace=True)
    mapping = dict(df[['CLUSTER', 'REGION']][df['REGION'].notna()].values)
    df['REGION'] = df['REGION'].fillna(df['CLUSTER'].map(mapping))
    df['CLUSTER']=df.groupby('REGION')['CLUSTER'].transform(lambda x: x.fillna(x.mode()[0]))
    df['YELLOW_PHASE_LOADING']=df.groupby('CLUSTER')['YELLOW_PHASE_LOADING'].transform(lambda x:x.fillna(x.mean()))
    df['RED_PHASE_LOADING']=df.groupby('CLUSTER')['RED_PHASE_LOADING'].transform(lambda x:x.fillna(x.mean()))
    df['BLUE_PHASE_LOADING']=df.groupby('CLUSTER')['BLUE_PHASE_LOADING'].transform(lambda x:x.fillna(x.mean()))
    df["Transformer_Repaired_By_TSW"].replace({"yes": "YES", "no": "NO","No":"NO","Yes":"YES","SCRAPPED":"YES"}, inplace=True)
    df['Transformer_Repaired_By_TSW']=df['Transformer_Repaired_By_TSW'].fillna('NO')
    df.replace({'Transformer_Repaired_By_TSW':{'-': 'NO'}},inplace=True)
    df['Max._oil_level_(Ltr.)']=pd.to_numeric(df['Max._oil_level_(Ltr.)'].mask((df['Max._oil_level_(Ltr.)'] == 'OK')|\
                                                                          (df['Max._oil_level_(Ltr.)'] == 'DIRECT OFFER') |\
                                                                          (df['Max._oil_level_(Ltr.)'] == '-')))
    df['Max._oil_level_(Ltr.)']=df.groupby('Rated_Power')['Max._oil_level_(Ltr.)'].transform(lambda x: x.fillna(x.mean()))
    #df.drop(columns=['Issuance_Date','FDF_ID.','Fault_Location'],inplace=True)
    df['Oil_Quantity']=pd.to_numeric(df['Oil_Quantity'].mask((df['Oil_Quantity']=='NIL') |\
                                                        (df['Oil_Quantity'] == '-') |\
                                                        (df['Oil_Quantity'] == 'A') |\
                                                        (df['Oil_Quantity'] == 'POM') |\
                                                        (df['Oil_Quantity'] == 'PM') |\
                                                        (df['Oil_Quantity'] == '35 (cooling pipe damage)') |\
                                                        (df['Oil_Quantity'] == 'pm')))
    df['Oil_Quantity']=df['Oil_Quantity'].fillna(0)
    df['Oil_%']=pd.to_numeric(df['Oil_%'].mask(df['Oil_%']=='-'))
    df['Oil_%'][df['Oil_%']>50]=0.53
    df['Oil_%'][df['Oil_%']>3]=1.4
    df['Oil_%'].fillna(df['Oil_Quantity']/df['Max._oil_level_(Ltr.)'],inplace=True)
    #df['Oil_%']=df['Oil_Quantity']/df['Max._oil_level_(Ltr.)']
    df.replace({'TTR_Status':{'3/FAULTY':'3,FAULTY','R/FAULTY':'R,FAULTY','Y/FAULTY':'Y,FAULTY',\
                         'B/FAULTY':'B,FAULTY','ok':'OK','3,FAULT':'3,FAULTY','3/Faulty':'3,FAULTY',\
                         'R,FAULT':'R,FAULTY','Y,FAULT':'Y,FAULTY','R,FAULYT':'R,FAULTY','y/faulty':'Y,FAULTY',\
                         'TTR NOT SUPPORTED':'3,FAULTY','OK,B/P':'OK','OK,HEAT-UP':'OK','3/faulty':'3,FAULTY',\
                         'OK,LOSSES':'OK','B,FAULT':'B,FAULTY','Y, FAULTY':'Y,FAULTY','Y/B/FAULTY':'Y,B,FAULTY',\
                         'ok, heatup':'OK','OK,BUT LOSSES':'OK','B/faulty':'B,FAULTY','ok,core unbalanced':'OK'\
                         }},inplace=True)
    df.replace({'TTR_Status':{'R/faulty':'R,FAULTY','OK,HEATUP & B/P':'OK','D':np.nan,'A':np.nan,'OK, HEAT UP':'OK',\
                         'HEATUP':'OK','B/P,OK':'OK','B/FAULTY,HEATUP':'B,FAULTY','Y,B, FAULTY':'Y,B,FAULTY',\
                         'R/Y/FAULTY':'R,Y,FAULTY','B/Faulty':'B,FAULTY','B, FAULTY':'B,FAULTY','3/F':'3,FAULTY',\
                         'R, FAULTY':'R,FAULTY','OK,B.P':'OK','TT NOT SUPPORTED':'3,FAULTY','ok,heatup':'OK',\
                         'Y/B FAULTY':'Y,B,FAULTY','?':np.nan,'OK,T.FAILED':'OK','ok,heatup/bp':'OK','OK- Heat up':'OK',\
                         ' OK':'OK'}},inplace=True)
    df.replace({'TTR_Status':{'OK,HEATUP':'OK','B/R/FAULTY':'R,B,FAULTY','Ok':'OK','OK HEATUP':'OK','Y&B,FAULTY':'Y,B,FAULTY',\
                         'R&B,FAULT':'R,B,FAULTY','R/B,FAULTY':'R,B,FAULTY','OK,BP':'OK','TTR FAIL':'3,FAULTY',\
                         'OK, Heat up':'OK','3, FAULTY':'3,FAULTY','Y, Faulty':'Y,FAULTY','3, Faulty':'3,FAULTY',\
                         'R, Faulty':'R,FAULTY','Y/B Faulty':'Y,B,FAULTY','Y/B,FAULTY':'Y,B,FAULTY','OK,LT ROD MELTED':'OK',\
                         'ok(testing fail)':'OK','ok.B/P':'OK','ok,B/P':'OK','B/FAULTY,CORE LOSSES':'B,FAULTY',\
                         'OK (TESTING FAILED JOB)':'OK','Y,B FAULTY':'Y,B,FAULTY','OK(TESTING LOSSES)':'OK',\
                         'R/B FAULTY':'R,B,FAULTY','R/YFAULTY':'R,Y,FAULTY','B/ LT FAULTY':'B,FAULTY','Y & B,FAULTY':'Y,B,FAULTY',\
                         'Y/R FAULTY':'R,Y,FAULTY'}},inplace=True)
    df.replace({'TTR_Status':{'-':np.nan,'R&Y FAULTY':'R,Y,FAULTY','R/Faulty':'R,FAULTY','OK,CORE U/B':'OK','R/Y FAULTY'\
                              :'R,Y,FAULTY','03,HEATUP':'OK','3 PHASE OPEN':'3,OPEN','Y, R, FAULTY':'R,Y,FAULTY'}},inplace=True)
    df.replace({'Transformer_being_removed_on':{'OTHERS,REINFORCEMENT/ DEINFORCEMENT / SIP':'REINFORCEMENT/ DEINFORCEMENT / SIP',\
                                           'DTS IN KH-E-GHAZI PUMPING FEEDER':np.nan,'REINFORCEMENT  OF TRANSFORMER':\
                                           'REINFORCEMENT/ DEINFORCEMENT / SIP','INTERNAL FAULT':'FAULT',\
                                           'ACCIDENT / FALLEN DOWN IN INCIDENT':'ACCIDENT CASE','Reinforcement':\
                                           'REINFORCEMENT/ DEINFORCEMENT / SIP','Fault':'FAULT','FAULTY':'FAULT'}},inplace=True)
    df.replace({'Transformer_being_removed_on':{'ACCIDENT CASE':'OTHER','Other':'OTHER','LINK BLOWN OUT':'FAULT','NOT INSTALLED':\
                                           'OTHER','COMPONENT DAMAGED':'FAULT','L.T BLUE PHASE BUSHING DAMAGE':\
                                           'PROACTIVE MAINTENANCE'}},inplace=True)
    df.replace({'Transformer_being_removed_on':{'-':np.nan}},inplace=True)
    df['Final_Conclusion'] = df['Final_Conclusion'].apply(clean_up)
    
    df['Date_of_Transformer_Approved']=pd.to_datetime(df['Date_of_Transformer_Approved']).dt.to_period('M')
    
    #df.drop(columns='Date_of_Transformer_Receipt',inplace=True)
    df['Work_Details_Carried_Out']=df['Work_Details_Carried_Out'].apply(lambda x: clean_up_work_details(str(x)))
    df['TTR_Status']=df['TTR_Status'].fillna(df['Work_Details_Carried_Out'])
    df.replace({'TTR_Status':{'RYB, HT & LT CHANGED':'3,FAULTY','Minor Repair':'OK','Partial Winding changed':'R,FAULTY',\
                         'RED PHASE HT & LT CHANGED':'R,FAULTY',np.nan:'OK'}},inplace=True)
    df['DTS_ID'] = df['DTS_ID'].map(lambda x: str(x).lstrip(r'[DdTtSs-]'))
    df.replace({'DTS_ID':{'nan':'missing','':'missing'}},inplace=True)
    df['Faulty_Transformer_Date']=pd.to_datetime(df['Faulty_Transformer_Date'],errors='coerce').dt.to_period('M')
    df['Faulty_Transformer_Date']=df['Faulty_Transformer_Date'].dt.strftime('%Y-%m')
    df['Faulty_Transformer_Date'].fillna('missing',inplace=True)
    df['Observations']=df['Observations'].apply(lambda x: clean_up_obs(str(x)))
    df.replace({'Make':{'T-PAK':'T/PAK','SIEMNES':'SIEMENS','T-POWER':'T/POWER','T.POWER':'T/POWER','Climax':'CLIMAX',\
                   'T.PAK':'T/PAK','Siemens':'SIEMENS','PAN/POWER':'P/POWER','ELEMETEC':'ELMETEC','Pel':'PEL','ELEMTEC':\
                   'ELMETEC','SIMENES':'SIEMENS','T-Pak':'T/PAK','PANPOWER':'P/POWER','VALIDAS':'VALIDUS','T/Power':'T/POWER',\
                   'PAN-POWER':'P/POWER','P-POWER':'P/POWER','P.POWER':'P/POWER','PAN.POWER':'P/POWER','T-Power':'T/POWER',\
                   'T.FAB':'T/FAB','Samco':'SAMCO','T/Pak':'T/PAK','T-FAB':'T/FAB','PAN POWER':'P/POWER',' CLIMAX':'CLIMAX',\
                   'PEL (PMU)':'PEL','SIEMENES':'SIEMENS',"SIEMENS'":"SIEMENS",'TFAB':'T/FAB','T-power':'T/POWER','T/POWER ':\
                   'T/POWER','Validus':'VALIDUS','pel':'PEL','TRANSPAK':'T/PAK','ELEMETC':'ELMETEC','  CLIMAX':'CLIMAX',\
                   "T/PAK'":"T/PAK",'Elemetch':'ELMETEC'}},inplace=True)
    df.replace({'Make':{'\s+':''}},regex=True,inplace=True)
    df.replace({'Make':{'PROTO':'missing','PMU':'missing','AHMED':'missing',';4':'missing'}},inplace=True)
    df['Faulty_Transformer_Date']=df['Faulty_Transformer_Date'].apply(lambda x: x[-2:])
    df.replace({'Faulty_Transformer_Date':{'ng':'missing'}},inplace=True)
    df['REMARKS']=df['REMARKS'].apply(lambda x: clean_up_remarks(str(x)))
    #df.replace({'REMARKS':{'FDF N/A':'FDF NA','-':np.nan,'nan':np.nan,'Ok Transformer':'OTHER'}},inplace=True)
    df['Transformer_being_removed_on']=df['Transformer_being_removed_on'].fillna(df['REMARKS'])
    df.replace({'Transformer_being_removed_on':{'LINK BLOWN OUT':'FAULT','OIL LEAKAGE':'PROACTIVE MAINTENANCE',\
                                            'OTHERS,REINFORCEMENT/ DEINFORCEMENT / SIP':'REINFORCEMENT/ DEINFORCEMENT / SIP',\
                                            'BUSHING DAMAGED':'FAULT','PHASE MISSING':'FAULT',\
                                            'RECONSTRUCTION OF S/S / ILLEGAL / LIFTED':'OTHER','INTERNAL FAULT':'FAULT',\
                                            'VARIANCE IN VOLTAGE':'FAULT','OTHERS':'OTHER','ABNORMAL / HUMMING SOUND':\
                                            'PROACTIVE MAINTENANCE','FEEDER TRIPPED':'FAULT',\
                                            'REPLACEMENT OF FAULTY PMT':'OTHER','COOLING PIPE DAMAGED':\
                                            'PROACTIVE MAINTENANCE'}},inplace=True)
    df['Avg_loading_per_phase']=df[['YELLOW_PHASE_LOADING','RED_PHASE_LOADING','BLUE_PHASE_LOADING']].mean(axis=1)
    df.replace({'Oil_status':{'-':np.nan,' C':'C','N':np.nan}},inplace=True)
    df['Oil_status']=df['Oil_status'].fillna(df['Final_Conclusion'])
    df.replace({'Oil_status':{'Loading Issue':'B','Other':'B','Proactive Maintenance':'D','Low Oil Level':'B',\
                         'Oil Leakage':'B','External Fault':'C','Maintenance Issues':'D','Aged Winding':'B'}},inplace=True)

    df.replace({'Tap_changer_status':{np.nan:'A','-':'Not Available','C':'Not Available','A-LEAKAGE':'A'}},inplace=True)
    df.replace({'Conservator_Tank_Valve':{'-':'Not Available','A-LEAKAGE':'OPEN-LEAKAGE','A-DAMAGE':'DAMAGE','OPEN (LEAKAGE)':\
                                     'OPEN-LEAKAGE',' Not Available':'Not Available',"Not Available`":"Not Available",\
                                     "`Not Available":"Not Available",'OPEN- LEAKAGE':'OPEN-LEAKAGE','A-RUSTED':'DAMAGE',\
                                     'DAMAGE-OPEN':'DAMAGE','CLOSED-LEAKAGE':'CLOSED','LEAKAGE':'OPEN-LEAKAGE','CLOSED-DAMAGE':\
                                      'CLOSED'}},inplace=True)
    df.replace({'Conservator_Tank_Valve':{np.nan:'A','D':'DAMAGE','B':'OPEN-LEAKAGE'}},inplace=True)
    df.replace({'Silica_Gel_Status':{'-':'Not Available',np.nan:'B'}},inplace=True)
    
    df.replace({'Top_Status':{'-':np.nan,'`-':np.nan}},inplace=True)
    df['Top_Status']=df['Top_Status'].fillna(df['Final_Conclusion'])
    df.replace({'Top_Status':{'Loading Issue':'B','Low Oil Level':'B','Other':'B','Proactive Maintenance':'D',\
                         'External Fault':'C','Oil Leakage':'B','Aged Winding':'B','Maintenance Issues':'D',\
                         'Internal/TSW Fault':'B'}},inplace=True)
    df.replace({'Tap_changer_status':{'A':1,'Not Available':0},\
           'Oil_status':{'A':1,'B':2,'C':3,'D':4},\
           'Silica_Gel_Status':{'Not Available':0,'A':1,'B':2,'C':3,'D':4},\
           'Top_Status':{'A':1,'B':2,'C':3,'D':4}},inplace=True)
    to_drop=['WO_No', 'Date_of_Transformer_Receipt', 'Date_of_Transformer_Approved','Transformer_Number'\
         ,'FDF_ID.','Issuance_Date', 'DTS_ID','Fault_Location','RED_PHASE_LOADING', 'YELLOW_PHASE_LOADING', \
         'BLUE_PHASE_LOADING','Average_laoding_greater_than_max_flag','Rated_Power','Max._oil_level_(Ltr.)', 'Oil_Quantity']
    df.drop(to_drop,axis=1, inplace=True)
    return df

def clean_up(final_conc):
    """Function to clean up target"""
    if re.search(r'TSW|tsw|[Ii]nternal|INTERNAL',final_conc):
        final_conc='Internal/TSW Fault'
    elif re.search(r'[Ii]nsulation|INSULATION|[wW]inding|WINDING',final_conc):
        final_conc='Aged Winding'
    elif re.search(r'[Ss]tock|STOCK|[Oo]ther|OTHER|[Aa]ccident|ACCIDENT|[Mm]issing|MISSING',final_conc):
        final_conc='Other'
    elif re.search(r'[lL]eakage|LEAKAGE',final_conc):
        final_conc='Oil Leakage'
    elif re.search(r'[eE]xternal|EXTERNAL|[sS]ystem|SYSTEM|[Pp]hase|PHASE',final_conc):
        final_conc='External Fault'
    elif re.search(r'[mM]aintenance|MAINTENANCE|[Ll]oose|LOOSE|[Bb]ush|BUSH|[Ff]lash|FLASH|[Ss]ilic|SILIC|[Tt]op|TOP|[Bb]rok|\
                    BROK|[Hh]opeles|HOPELES|[Cc]lamp|CLAMP|[Pp]oor|POOR',final_conc):
        final_conc='Maintenance Issues'
    elif re.search(r'[Ll]evel|LEVEL',final_conc):
        final_conc='Low Oil Level'
    elif re.search(r'PM|[Pp]rocative|[Pp]reventive|PROACTIVE|PREVENTIVE', final_conc):
        final_conc='Proactive Maintenance'
    else:
        final_conc='Loading Issue'
    return final_conc

def clean_up_work_details(work_details):
    """Function to clean up work details"""
    if re.search(r'[Mm]inor|MINOR',work_details):
        work_details='Minor Repair'
    elif re.search(r'RYB|R Y B|R,Y,B|3|CAN',work_details):
        work_details='RYB, HT & LT CHANGED'
    elif re.search(r'[Rr]ed|RED|R HT|R ph',work_details):
        work_details='RED PHASE HT & LT CHANGED'
    elif re.search(r'[Bb]lue|BLUE|B HT|B \(HT',work_details):
        work_details='BLUE PHASE HT & LT CHANGED'
    elif re.search(r'[Yy]ellow|YELLOW|Y HT|Y,HT|Y,BH|^YELL',work_details):
        work_details='YELLOW PHASE HT & LT CHANGED'
    elif re.search(r'1|2|[Pp]artial|PARTIAL|One|^LT',work_details):
        work_details='Partial Winding changed'
    return work_details

def clean_up_obs(obs):
    """Function to clean up observations"""
    if re.search(r'[Tt]op|TOP',obs) and re.search(r'[Bb]ushings|HT|LT',obs) and re.search(r'[Ll]ow|LOW|[Ll]evel|LEVEL',obs):
        obs='Top poor, HT & LT bushings hopeless, Low oil level'
    elif re.search(r'[Tt]op|TOP',obs) and re.search(r'[Bb]ushings|HT|LT',obs):
        obs='Top poor, HT & LT bushings hopeless'
    elif re.search(r'[Tt]op|TOP',obs) and re.search(r'[Ll]ow|LOW|[Ll]evel|LEVEL',obs):
        obs='Top poor, Low oil level'
    elif re.search(r'[Ll]ow|LOW|[Ll]evel|LEVEL',obs) and re.search(r'[Oo]verloading|OVERLOADING|[Uu]nbalancing',obs):
        obs='Low oil level, Overloading'
    elif re.search(r'[Ll]ow|LOW|[Ll]evel|LEVEL',obs) and re.search(r'[Bb]ushings|HT|LT',obs):
        obs='Low oil level, Flashes on bushing'
    elif re.search(r'[Ll]ow|LOW|[Ll]evel|LEVEL',obs) and re.search(r'[Aa]ged|[Ww]inding|[Ii]nsulation',obs):
        obs='Low oil level, Aged insulation'
    elif re.search(r'[Oo]verloading|OVERLOADING|[Uu]nbalancing',obs):
        obs='Overloading'
    elif re.search(r'[Cc]ooling|COOLING',obs):
        obs='Cooling pipe damaged'
    elif re.search(r'[Ll]ow|LOW|[Ll]evel|LEVEL',obs):
        obs='Low oil level'
    elif re.search(r'[Aa]ged|[Ww]inding|[Ii]nsulation',obs):
        obs='Aged winding'
    elif re.search(r'[Tt]op|TOP',obs):
        obs='Top poor'
    else:
        obs='FDF NA (Fault cannot be determined)'
    return obs



def clean_up_remarks(remarks):
    if re.search(r'[Ll]ink|LINK|LInk|[Bb]urn|BURN',remarks):
        remarks='LINK BLOWN OUT'
    elif re.search(r'[Ll]eakage|LEAKAGE|[Dd]rain|DRAIN',remarks):
        remarks='OIL LEAKAGE'
    elif re.search(r'[Ff][Dd][Ff] [Nn][ //][Aa]',remarks):
        remarks='FDF NA'
    elif re.search(r'[Ii]nternal|INTERNAL', remarks) and re.search(r'[Ff]ault|FAULT',remarks):
        remarks='INTERNAL FAULT'
    elif re.search(r'[Pp]hase|PHASE',remarks) and re.search(r'[Mm]iss|MISS',remarks):
        remarks='PHASE MISSING'
    elif re.search(r'[Cc]ompo|COMPO',remarks) and re.search(r'[Dd]amag|DAMAG',remarks):
        remarks='COMPONENT DAMAGED'
    elif re.search(r'[Oo]ther|OTHER|[Mm]egg|MEGG|[Oo]ld|OLD',remarks):
        remarks='OTHERS'
    elif re.search(r'[Aa]bnorm|ABNORM|[Hh]umm|HUMM',remarks) and re.search(r'[Ss]oun|SOUN|[Nn]ois|NOISE',remarks):
        remarks='ABNORMAL / HUMMING SOUND'
    elif re.search(r'[Aa]ccid|ACCID',remarks):
        remarks = 'ACCIDENT / FALLEN DOWN IN INCIDENT'
    elif re.search(r'[Pp]roact|PROACT|[Mm]aint|MAINT',remarks):
        remarks = 'PROACTIVE MAINTENANCE'
    elif re.search(r'[Vv]olt|VOLT|[Vv]arian|VARIAN',remarks):
        remarks = 'VARIANCE IN VOLTAGE'
    elif re.search(r'[Rr]einf|REINF|Re-inf|RE-INF|[Rr]e-enf|Re-Inf|RE-ENF|DE-INF|DEINF|DE-ENF|[Dd]einf|De-inf|De-Inf|De-Enf',remarks):
        remarks='OTHERS,REINFORCEMENT/ DEINFORCEMENT / SIP'
    elif re.search(r'[Rr]eplac|REPLACE',remarks):
        remarks = 'REPLACEMENT OF FAULTY PMT'
    elif re.search(r'[Cc]onserv|CONSERV',remarks):
        remarks='CONSERVTOR TANK ISSUE'
    elif re.search(r'[Bb]ush|BUSH|[Pp]has|PHAS',remarks):
        remarks='BUSHING DAMAGED'
    elif re.search(r'[Cc]ool|COOL',remarks):
        remarks='COOLING PIPE DAMAGED'
    elif re.search(r'[Tt]rip|TRIP|[Ff]eed|FEED',remarks):
        remarks='FEEDER TRIPPED'
    else:
        remarks='MISSING/UNINTELLIGIBLE'
        
    return remarks

def get_num_feats(df):
    num_feats = df.dtypes[df.dtypes != 'object'].index.tolist()
    return num_feats

def get_cat_feats(df):
    cat_feats = df.dtypes[df.dtypes == 'object'].index.tolist()
    return cat_feats

def get_train_test(df):
    y=df.Final_Conclusion
    df.drop('Final_Conclusion',axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2,random_state = 0)
    return X_train, X_test, y_train, y_test