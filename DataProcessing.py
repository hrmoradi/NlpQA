from Libraries import *


def extractPS_CR(question, patID, CPTCode, rawText):
    #     PS_text = r"(\bps\b)|(\bp\/s\b)|(stabilized posterior)|(posterior stabilized)|(cruciate ligaments were removed)"
    #     CR_text = r"(\bcr\b)|(\bc\/r\b)|(cruciate retaining)"

    PS_text = re.compile(r"(\bps\b)|(\bp\/s\b)|"
                         "(stabilized posterior)|(posterior stabilized)|(post stabilized)|"
                         "(cruciate ligaments were removed)|(posterior cruciate was excised)|"
                         "(tib ins stab)|(tib insert stab)|(posterior stabilized insert)|"
                         "(cruciate holes made)|(posterior cruciate substituted)|(\bpoly stab\b)|(\bpost stab\b)|"
                         "(PCL sacrificed)|(cruciate substituted)|(box cuts)|(post cruciate substituting)|"
                         "(PCL excised)|(cruciate ligaments excised)|(cruciate ligaments removed)|"
                         "(rotating platform)", re.IGNORECASE)

    CR_text = re.compile(r"(\bcr\b)|(\bc\/r\b)|(cruciate retaining)|(fixed bearing insert)|(PCL retained)"
                         "(posterior cruciate was intact and retained)|(cruciate retaining femoral component)|"
                         "\bcruc ret\b", re.IGNORECASE)

    if PS_text.search(rawText):
        rslt = PS_text.search(rawText)
        lbl_start = rslt.span()[0]
        lbl_end = rslt.span()[1]
        raw_label = rslt.group(0)
        label = "PS"
        raw_lbl_start = rslt.span()[0]
        raw_lbl_end = rslt.span()[1]
    elif CR_text.search(rawText):
        rslt = CR_text.search(rawText)
        lbl_start = rslt.span()[0]
        lbl_end = rslt.span()[1]
        raw_label = rslt.group(0)
        label = "CR"
        raw_lbl_start = rslt.span()[0]
        raw_lbl_end = rslt.span()[1]
    else:
        label = ""
        raw_label = ""
        lbl_start = ""
        lbl_end = ""
        raw_lbl_start = ""
        raw_lbl_end = ""

    return [patID, CPTCode, question, label, lbl_start, lbl_end, raw_label, raw_lbl_start, raw_lbl_end]


def extractMakerImplant(question, patID, CPTCode, rawText):
    #     textList = r"""(smith and nephew legion)|(depuy attune)|(depuy sigma)|(depuy lcs)|(depuy lps)|
    #     (smith and nephew hinge)|(posterior stabilized).*?(djo)|(djo).*?(posterior stabilized)|
    #     (conformis itotal)|(djo empowr)|(stanmore jts)|(stryker triathalon)|
    #     (wright medical repiphysis)|(wright medical guardian)|(zimmer persona)"""

    #     textList = r"""(smith and nephew legion)|(depuy attune)|(depuy sigma)|(depuy lcs)|(depuy lps)|
    #     (smith and nephew hinge)|
    #     (conformis itotal)|(djo empowr)|(stanmore jts)|(stryker triathalon)|
    #     (wright medical repiphysis)|(wright medical guardian)|(zimmer persona)"""

    textList = r"""(smith and nephew legion)|(depuy attune)|(depuy sigma)|(conformis)|(djo)"""

    search_text = re.compile(textList, re.IGNORECASE)
    rslt = re.search(search_text, rawText)

    if rslt:
        lbl_start = rslt.span()[0]
        lbl_end = rslt.span()[1]
        raw_label = rslt.group(0)
        label = rslt.group(0)
        raw_lbl_start = rslt.span()[0]
        raw_lbl_end = rslt.span()[1]
    else:
        label = ""
        raw_label = ""
        lbl_start = ""
        lbl_end = ""
        raw_lbl_start = ""
        raw_lbl_end = ""

    return [patID, CPTCode, question, label, lbl_start, lbl_end, raw_label, raw_lbl_start, raw_lbl_end]


def extractLaterality(question, patID, CPTCode, rawText):
    bilat_text = r"""(bilateral knee replacement)|(bilateral knee replacements)|
    (bilateral total knee replacement)|(bilateral total knee replacements)|
    (bilateral knee arthroplasty)|(bilateral total knee arthroplasty)"""

    left_text = r"""(left knee replacement)|(left total knee replacement)|
    (left knee arthroplasty)|(left total knee arthroplasty)|
    (left complex total knee arthroplasty)|
    (left total replacement)|(left complex total knee replacement)|
    (left distal total oncologic knee replacement)
    (left distal femur replacement and total oncologic knee)|
    (left distal femur replacement/total oncologic knee replacement)|
    (left distal femur replacement - total oncologic knee replacement)|
    (left distal femur total oncologic knee replacement)|
    (left total oncologic knee replacement)|
    (left knee with total knee replacement)|(left TKR)|
    (left total knee)|(left conformis total knee arthroplasty)|
    (left conformis total knee)|(conformis total knee arthroplasty left knee)"""

    right_text = r"""(right knee replacement)|(right total knee replacement)|
    (right knee arthroplasty)|(right total knee arthroplasty)|
    (right complex total knee arthroplasty)|
    (right total replacement)|(right complex total knee replacement)|
    (right distal total oncologic knee replacement)|
    (right distal femur replacement and total oncologic knee)|
    (right distal femur replacement/total oncologic knee replacement)|
    (right distal femur replacement - total oncologic knee replacement)|
    (right distal femur total oncologic knee replacement)|
    (right total oncologic knee replacement)|
    (right knee with total knee replacement)|(right TKR)|
    (right total knee)|(right conformis total knee arthroplasty)|
    (right conformis total knee)|(conformis total knee arthroplasty right knee)"""

    search_text = re.compile(fr"{left_text}|{right_text}|{bilat_text}", re.IGNORECASE)

    if search_text.search(rawText):
        rslt = search_text.search(rawText)
        # raw label aka such as 'right knee replacement'
        raw_lbl_start = rslt.span()[0]
        raw_lbl_end = rslt.span()[1]
        raw_label = rslt.group(0)

        # true label aka 'right'
        if re.search("left", rslt.group(0), re.IGNORECASE):
            rslt2 = re.search("left", rslt.group(0), re.IGNORECASE)
            lbl_start = raw_lbl_start + rslt2.span()[0]
            lbl_end = raw_lbl_start + rslt2.span()[0] + rslt2.span()[1]
            label = rslt2.group(0)
        elif re.search("right", rslt.group(0), re.IGNORECASE):
            rslt2 = re.search("right", rslt.group(0), re.IGNORECASE)
            lbl_start = raw_lbl_start + rslt2.span()[0]
            lbl_end = raw_lbl_start + rslt2.span()[0] + rslt2.span()[1]
            label = rslt2.group(0)
        elif re.search("bilateral", rslt.group(0), re.IGNORECASE):
            rslt2 = re.search("bilateral", rslt.group(0), re.IGNORECASE)
            lbl_start = raw_lbl_start + rslt2.span()[0]
            lbl_end = raw_lbl_start + rslt2.span()[0] + rslt2.span()[1]
            label = rslt2.group(0)

    else:
        label = ""
        raw_label = ""
        lbl_start = ""
        lbl_end = ""
        raw_lbl_start = ""
        raw_lbl_end = ""

    return [patID, CPTCode, question, label, lbl_start, lbl_end, raw_label, raw_lbl_start, raw_lbl_end]


def extractPatella(question, patID, CPTCode, rawText):
    postextList = r"""((?:(round|oval|dome)).+?(?:patella|patellar))|((?:patella|patellar) button)|
    ((?:patella|patellar) resurfacing)|(resurfaced (?:patella|patellar))|
    ((?:patella|patellar) (?:was|were) prepared)|((?:patella|patellar) (?:was|were) cemented)|
    ((?:patella|patellar) component)|(surface of (?:patella|patellar) removed)|
    ((?:patella|patellar) surface (?:was|were) osteotomized)|
    ((?:patella|patellar) was resurfaced)|((?:patella|patellar) was cut)|
    ((?:patella|patellar) ream)"""

    negtextList = r"""(without (?:patella|patellar) resurfacing)|
    (unresurfaced (?:patella|patellar))|((?:patella|patellar) unresurfaced)|
    ((?:patella|patellar) (?:was|were) not)|(not to resurface)"""

    search_text_neg = re.compile(negtextList, re.IGNORECASE)
    search_text_pos = re.compile(postextList, re.IGNORECASE)
    rslt_neg = re.search(search_text_neg, rawText)
    rslt_pos = re.search(search_text_pos, rawText)

    if rslt_neg:
        lbl_start = ""
        lbl_end = ""
        label = "No patella resurfacing"
        raw_label = rslt_neg.group(0)
        raw_lbl_start = rslt_neg.span()[0]
        raw_lbl_end = rslt_neg.span()[1]
    elif rslt_pos:
        lbl_start = ""
        lbl_end = ""
        label = "Patella resurfacing"
        raw_label = rslt_pos.group(0)
        raw_lbl_start = rslt_pos.span()[0]
        raw_lbl_end = rslt_pos.span()[1]
    # In this case, no regex hits indicate no resurfacing
    else:
        label = "No patella resurfacing"
        raw_label = ""
        lbl_start = ""
        lbl_end = ""
        raw_lbl_start = ""
        raw_lbl_end = ""

    return [patID, CPTCode, question, label, lbl_start, lbl_end, raw_label, raw_lbl_start, raw_lbl_end]


# This function currently filters out the labels with small sample sizes
def filterSamples(inputNotes):
    # Remove all Laterality Questions that have 'bilateral' label
    medicalLabels = medicalLabels.loc[(medicalLabels["Question"] == "What is the laterality?") &
                                      (medicalLabels["Label"].str.lower() != "bilateral")]

# This function modifies the OP Notes and replaces 'post stabilized' and 'ps' to 'posterior stabilized'
# and replaces 'cr' with 'cruciate retaining'
def changeOPNote(rawText):
    if re.search(r"(post stabilized)", rawText, re.IGNORECASE):
        compiled = re.compile(r"post stabilized", re.IGNORECASE)
        rawText = compiled.sub("posterior stabilized", rawText)
    #         rawText = re.sub("post stabilized", "posterior stabilized", rawText, flags=re.IGNORECASE)

    if re.search(r"(\bps\b)", rawText, re.IGNORECASE):
        compiled = re.compile(r"\bps\b", re.IGNORECASE)
        rawText = compiled.sub("posterior stabilized", rawText)
    #         rawText = re.sub("\bps\b", "posterior stabilized", rawText, flags=re.IGNORECASE)

    if re.search(r"(\bcr\b)", rawText, re.IGNORECASE):
        compiled = re.compile(r"\bcr\b", re.IGNORECASE)
        rawText = compiled.sub("cruciate retaining", rawText)
    #         rawText = re.sub("\bcr\b", "cruciate retaining", rawText, flags=re.IGNORECASE)

    return rawText


def constructData(dataPath, fileType, data_details):
    if fileType == "smaller":
        if data_details["scale_back"] == 2:
            print("Using original file...")
            xlsxFileName = "R521_27447_OP_NOTE_102(Original).XLSX"
        else:
            print("Using corrected file...")
            xlsxFileName = "R521_27447_OP_NOTE_102(Simplified).XLSX"
    elif fileType == "larger":
        if data_details["scale_back"] == 2:
            print("Using original file...")
            xlsxFileName = "TOTAL_KNEE_ARTHROPLASTY__(27447)(Original except for 2 lines).XLSX"
        else:
            print("Using corrected file...")
            xlsxFileName = "TOTAL_KNEE_ARTHROPLASTY__(27447)(Simplified).XLSX"

    xlsx_file_path = os.path.join(dataPath, xlsxFileName)

    medNotes_dtypes = {"OP_NOTE": str, "AGE at CPT CODE": 'Int64', "height in Inches": 'Float64',
                       "Weight in KGs": 'Float64',
                       "Last recorded BMI": 'Float64', "Ethnic_Group": str, "Smoking": str, "Sex": str, "Race": str}

    # Read in medical notes
    medicalNotes = pd.read_excel(xlsx_file_path, dtype=medNotes_dtypes, na_values="NULL")

    # Drop rows that are all missing
    medicalNotes = medicalNotes.dropna(axis=0, how="all")
    medicalNotes = medicalNotes.reset_index(drop=True)

    # Remove any extraneous spaces
    medicalNotes["OP_NOTE"] = medicalNotes["OP_NOTE"].apply(lambda x: " ".join(x.split()))

    if data_details["scale_back"] == 0:
        # Replace 'post stabilized', 'ps', and 'cr' with appropriate substitutions
        medicalNotes["OP_NOTE"] = medicalNotes["OP_NOTE"].apply(lambda x: changeOPNote(x))

    # Define label dataframe
    medicalLabels = pd.DataFrame(
        columns=["pat_id", "CPT Code Date", "Question", "Label", "Label_Start", "Label_Stop", "Raw_Label",
                 "Raw_Label_Start", "Raw_Label_Stop"], dtype='int64')

    # Add Implant Maker and Name Question
    medicalLabels = pd.concat(
        [medicalLabels, pd.DataFrame(medicalNotes.apply(lambda x: extractMakerImplant("What is the implant?",
                                                                                      x["pat_id"], x["CPT Code Date"],
                                                                                      x["OP_NOTE"]), axis=1).tolist(),
                                     columns=["pat_id", "CPT Code Date", "Question", "Label", "Label_Start",
                                              "Label_Stop", "Raw_Label", "Raw_Label_Start", "Raw_Label_Stop"])], axis=0)
    # Add Constraint Type Question
    medicalLabels = pd.concat(
        [medicalLabels, pd.DataFrame(medicalNotes.apply(lambda x: extractPS_CR("What is the constraint type?",
                                                                               x["pat_id"], x["CPT Code Date"],
                                                                               x["OP_NOTE"]), axis=1).tolist(),
                                     columns=["pat_id", "CPT Code Date", "Question", "Label", "Label_Start",
                                              "Label_Stop",
                                              "Raw_Label", "Raw_Label_Start", "Raw_Label_Stop"])], axis=0)

    # Add Laterality Question
    medicalLabels = pd.concat(
        [medicalLabels, pd.DataFrame(medicalNotes.apply(lambda x: extractLaterality("What is the laterality?",
                                                                                    x["pat_id"], x["CPT Code Date"],
                                                                                    x["OP_NOTE"]), axis=1).tolist(),
                                     columns=["pat_id", "CPT Code Date", "Question", "Label", "Label_Start",
                                              "Label_Stop", "Raw_Label", "Raw_Label_Start", "Raw_Label_Stop"])], axis=0)

    # If 4 questions specified, add patella question labels
    if str(data_details["num_questions"]) == "4":
        pat = pd.DataFrame(medicalNotes.apply(lambda x: extractPatella("Is there patella resurfacing?",
                                                                       x["pat_id"], x["CPT Code Date"],
                                                                       x["OP_NOTE"]), axis=1).tolist(),
                           columns=["pat_id", "CPT Code Date", "Question", "Label",
                                    "Label_Start", "Label_Stop", "Raw_Label",
                                    "Raw_Label_Start", "Raw_Label_Stop"])
        # Keep only raw labels that is less than 6 words
        pat = pat.loc[(pat["Raw_Label"].str.split().apply(lambda x: len(x)) < 6), :]

        if fileType == "larger":
            # Only keep labels that occur at least 100 or more times
            pat_lbl_cnts = pat["Raw_Label"].value_counts().loc[lambda x: x >= 100].index.tolist()
        elif fileType == "smaller":
            # Only keep labels that occur at least 20 or more times
            pat_lbl_cnts = pat["Raw_Label"].value_counts().loc[lambda x: x >= 20].index.tolist()
        pat = pat.loc[pat["Raw_Label"].isin(pat_lbl_cnts), :]

        pat.loc[pat["Raw_Label_Start"] == "", "Raw_Label_Start"] = pd.NA
        pat.loc[pat["Raw_Label_Stop"] == "", "Raw_Label_Stop"] = pd.NA
        pat.loc[pat["Label_Start"] == "", "Label_Start"] = pd.NA
        pat.loc[pat["Label_Stop"] == "", "Label_Stop"] = pd.NA
        pat = pat.astype({"Label_Start": "Int64", "Raw_Label_Start": "Int64"})
        pat = pat.convert_dtypes()
        medicalLabels = pd.concat([medicalLabels, pat])

    #####################################################################################################

    # Remove any bilateral labels from laterality
    # Remove any rotating platform from constraint type
    # Remove any of the below implant makers
    makers_to_remove = ["stanmore", "stryker", "wright medical", "zimmer"]
    medicalLabels = medicalLabels.loc[(medicalLabels["Label"].str.lower() != "bilateral") &
                                      (medicalLabels["Raw_Label"].str.lower() != "rotating platform") &
                                      (~medicalLabels["Label"].str.lower().isin(makers_to_remove))]

    # Drop rows (aka questions) that do not have an answer
    medicalLabels = medicalLabels.loc[medicalLabels["Label"] != "", :]
    medicalLabels = medicalLabels.reset_index(drop=True)

    medicalLabels = medicalLabels.astype({"pat_id": 'str', "Question": 'str', "Label": 'str', "Raw_Label": 'str',
                                          "Label_Start": 'Int64', "Label_Stop": 'str', "Raw_Label_Start": 'Int64',
                                          "Raw_Label_Stop": 'str'})

    medicalLabels["CPT Code Date"] = pd.to_datetime(medicalLabels["CPT Code Date"])
    medicalNotes = medicalNotes.merge(medicalLabels, on=["pat_id", "CPT Code Date"])

    return medicalNotes