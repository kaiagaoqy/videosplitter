## Run
Required Packages
```bash
pip install PyQt5 opencv-python pandas
```

run
```bash
python main.py
```

## Instructions
1. Load Video        
	•	Click on the Load Video button.    
	•	Select the video files you wish to process from your device.       

2. Add Tags

	•	Purpose: Label objects and their actions within the video.         
	•	Steps:
	1.	Enter Tag Name:
	  •	Input a descriptive name for the object (e.g., Chair_1m).
	2.	Select Action:
  	•	Choose the appropriate action from the dropdown menu:      
    	•	Start: Marks the beginning of the object’s appearance.      
    	•	Annotate: Indicates a specific annotation point between start and end frames.       
    	•	End: Marks the conclusion of the object’s appearance.                
	3.	Add Tag:
	  •	Click the Add Tag button to apply the tag with the specified name and action. 

3. Split Video

	•	Purpose: Divide the video into segments based on the Start and End tag pairs.         
	•	Steps:
  	1. Click on the `Split Video` button to split the video at each pair of Start and End tags, creating separate clips for each segment.

5. Save Tags

   
	•	Action: Export all tags to a CSV file for documentation and further analysis.      
	•	Steps:
	  1.	Click on the Save Tags or Export button.
	
 
 CSV Format Example:
   |Frame|	Time (s)|	Object Name|	Action|	Mask Path| HandTip|
   |:---|:---|:---|:---|:---|:---|
   |1	|0.04|	chair_1m|	Start|	masks/scenevideo/frame_1_mask.png| (823.0,170.0)|
   |24	|0.14|	chair_1m|	End|	masks/scenevideo/frame_24_mask.png| (823.0,170.0)|

<img width="1284" alt="image" src="https://github.com/user-attachments/assets/72b21068-a9bd-4cfb-bb06-fe8f92ef5b27">

5. Annotate Mask and HandTip

	•	Purpose: Precisely outline objects specific points (e.g., hand tips) within frames.   
	•	Steps:
	1.	Access Frame:
	•	Double-click an entry in the tag list to navigate to the corresponding frame in the video.
	2.	Annotate Polygon:
	•	Use the polygon tool to draw around the object, creating a mask that outlines its shape. then save the mask
	3.	Annotate HandTip:
	•	Select the point tool and click on the exact location of the hand tip or relevant point within the frame. then save the point

