# ImageFeeds
A common interface to control all cameras. 

### Supported Feeds  
- Live webcam (via OpenCV)
- Live IPCamera
- Video File (via Moviepy, Skvideo)
- RealSense
- Azure Kinect
- DSLRCamera (via GPhoto2)
- (WIP) Wraycam

Tested on Ubuntu 20.04

### Install
`python setup.py install`  
You will need to install the separated packages for respective cameras.
(Azure, RealSense, DSLR, Wraycam)


### Usage

```python
from ImageFeeds import CVFeed
import numpy as np

feed = CVFeed(display=True, #show the live result?
    thread=True, #threading to skip frames
    post_processing=[], #some functions to process the frame 
    )
feed.start()



img:np.array = feed.read() # Non-blocking image retrival
```