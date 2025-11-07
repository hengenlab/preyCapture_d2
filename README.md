# preyCapture_d2
collaboration project with the Shew lab


## Experiment structure

A mouse catches roaches. It gets 6 roaches per day. You can see the videos in <b>prey_capture_videos</b>

If a mouse catch the roach quick, we give them 1 roach every 10 minutes (a day of experiment can be done in 1hr)
<br>
If a mouse doesn't catch the roach, we give them 30-minute upper limit to catch a roach, then we swap a new roach in (that means a 3-hr experiment)

The updated time where the mouse get the roach (start_hunt) and caught the roach (end_hunt) are in <b>[this google sheet](https://docs.google.com/spreadsheets/d/1d95gF3JoxJ4COLAago3wvoaTeg8tu6OcpiFPSr4pBxM/edit?gid=102648672#gid=102648672)</b>. Less updated, but stable record of prey capture information are in the repo folder <u>prey-capture-record</u>

The <b>start_hunt</b> and <b>end_hunt</b> time are based on the timestamp in the 1-hour videos in <b>prey_capture_videos</b>

### Pre-trial
Before the formal 6 roach * 4 days experiment, the mouse also get a pre-trial, which happens 2 days before the first formal trial (mouse was not food deprived before pre-trial).

Mouse get 1 roach at 5pm on the pre-trial day. The mouse have unlimited time to catch that roach. We take out the pre-trial roach next morning (10am-ish) if the mouse did not hunt for the entire night.

### File notes

The file <b>PreyCapture_D2.ipynb</b> shows all analysis that I've done on this prey capture dataset.
<br>
I did it on the Hengen lab server, so the filepaths are a little different.

## Data Structure:

### Neural data
In Hengen Lab, each neural recording is put into a folder like this:
<br>
<b>RCC00036_L1_W2_2025-05-19_11-28-08</b>, where 2025-05-19 11:28:08 is the time we start the recording.
<br>
Each recording session usually last 1-3 days

The ephys data from the recording is sorted in 4hr blocks, so you will see files that looks like:
<br>
<b>H_2025-05-20_15-28-11_2025-05-20_19-23-11_neurons_group0.npy</b>
<br>
This filename means that it cotains the neural data from 2025-05-20 15:28:11 to 19:28:11
<br>
(each raw data file is 5 minutes, so the raw data file from 2025-05-20_19-23-11 ends at 19:28:11)

These 4 hour sorting blocks keeps going. A hunt session might contain 1 or 2 of these sorting blocks.

<b> [This google sheet](https://docs.google.com/spreadsheets/d/1rviEQ1JmN-5eOsBmWacZ4hrLSJz4KnDJG-m2IQKaaII/edit?gid=2072453467#gid=2072453467) </b> records all neural data for this project, and also keeps track of the data clean process.


### Videos
Video recordings always starts after the neural recordings.
<br>
Video files are formated like: [animal]\_[cage-type]\_[camera-name]\_[time-range]
<br>
(e.g. <b>RCC00036_bucket_e3v810a-20250522T083736-093737.mp4</b>)

In rcc_prey_capture_record.xlsx, you will see a row like this:
<pre>
trial  start_hunt  end_hunt   capture_time  Video_File
  1     0:40:28     0:53:09    0:12:41      RCC00036_bucket_e3v810a-20250522T083736-093737.mp4
</pre>

Given that the video starts at 16:37:06, the actual time of when the hunt starts is 08:37:36 + 0:40:28 = 09:38:04


### Combining videos with neural data
Continue with the hunt started at 2025-05-22 09:38:04. The alignment is done completely on wall time.

To see the video of the hunt: watch between 00:40:28 and 00:53:09 in the 1hr video [RCC00036_bucket_e3v810a-20250522T083736-093737.mp4]

To get neural data of the hunt: get spiketime between 02:09:52 and 02:22:33 in the 4hr sorting block  [H_2025-05-22_07-28-12_2025-05-22_11-23-12_neurons_group0.npy]

The spiketime start (02:09:52) is from 09:38:04 (hunt time, calculated in last secion) - 07:28:12 (sorting block start time)
<br>
The spiketime end (02:22:33) is from 08:37:36 (video start time) + 0:53:09 (video offset) - 07:28:12 (sorting block start time)
