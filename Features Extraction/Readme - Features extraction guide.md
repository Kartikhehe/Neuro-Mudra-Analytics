**Features Extraction guide** 

**Neuro Mudra Analytics**

At first, we need to change the counters going from 0 to 255 repetitively to time that starts from 0 seconds and keep on increasing. For that we need to run the following code for each mudra file:

**import pandas as pd**

**\# Load your CSV**  
**df \= pd.read\_csv("/Users/kartikraj/Desktop/EEG project files/Features extraction Rishitha/EEG\_CSVs/rishitha-yoni.csv")**

**\# Parameters**  
**sampling\_rate \= 256  \# Hz**  
**max\_counter \= 255**

**\# Reconstruct continuous sample indices**  
**sample\_indices \= \[\]**  
**cycle \= 0**

**for i in range(len(df)):**  
    **current \= df.loc\[i, 'Counter'\]**  
      
    **if i \> 0:**  
        **prev \= df.loc\[i \- 1, 'Counter'\]**  
        **if current \< prev:  \# Detect wraparound**  
            **cycle \+= 1**

    **absolute\_index \= current \+ cycle \* (max\_counter \+ 1\)**  
    **sample\_indices.append(absolute\_index)**

**\# Normalize: start from 0, convert to time**  
**min\_index \= sample\_indices\[0\]**  
**df\['Time'\] \= \[(i \- min\_index) / sampling\_rate for i in sample\_indices\]**

**\# Drop Counter column**  
**df \= df.drop(columns=\['Counter'\])**

**\# Move Time column to the front**  
**df \= df\[\['Time'\] \+ \[col for col in df.columns if col \!= 'Time'\]\]**

**\# Save (optional)**  
**df.to\_csv("/Users/kartikraj/Desktop/EEG project files/Features extraction Rishitha/EEG\_CSVs/rishitha-yoni.csv", index=False)**

**\# Show result**  
**print(df.head())**

The EEG\_generate\_training\_matrix.py script imports functions directly from EEG\_feature\_extraction.py. This means EEG\_feature\_extraction.py acts as a module that provides tools for EEG\_generate\_training\_matrix.py to use.

You only need to run the EEG\_generate\_training\_matrix.py script. When it runs, it will automatically use the functions defined in EEG\_feature\_extraction.py as needed.

To run the process:

Ensure both files are in the same directory.

Open your terminal or command prompt.

Navigate to the directory where you saved these two files and your EEG data CSVs.

Run the EEG\_generate\_training\_matrix.py script using a command like this:  
 python EEG\_generate\_training\_matrix.py \<input\_directory\_path\> \<output\_file\_name.csv\>

 /opt/anaconda3/bin/python EEG\_generate\_training\_matrix.py EEG\_CSVs extracted\_features.csv

In a folder, you must have a these two files: 

* EEG\_feature\_extraction.py  
* EEG\_generate\_training\_matrix.py

Also, you must a have a sub-folder in that folder only named as EEG\_CSVs, which must have all the mudra recorded biosignal data files named exactly as following (The name here can be changed e.g. kartik here):

* kartik-chandra.csv  
* kartik-kamal.csv  
* kartik-kanista.csv  
* kartik-mrigi.csv  
* kartik-mushti.csv  
* kartik-prana.csv  
* kartik-pranam.csv  
* kartik-samana.csv  
* kartik-vajra.csv  
* kartik-yoni.csv

Note that the name of the mudras should be written exactly same as above without any spelling error as the code concatenates the extracted files and labels the mudras from 0 to 9 based on the file names.

Implementing the above steps extracts 10 features from each channels and labels them from 0 to 9 based on the mudra.  
