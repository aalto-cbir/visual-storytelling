### Data Analysis:

Issues - 

1. The number of albums mentioned everywhere: `10,117`. Actual number of albums: **8031 + 998 + 1011 = 10040** from the data
2. The number of images mentioned everywhere: `210,819`. Actual number of images: **167528 + 21048 + 21075 = 209,651**
3. The json files contain `albums` information (that contains a field called `photos` indicating the number of images associated with that album). This is wrong. In actuality there are no albums in all `3` data splits (test, train, valid) that contain less than `10` images. But the json files show albums with `< 10` count. Example album-id - `616890`. More analysis in this [notebook](https://version.aalto.fi/gitlab/CBIR/visual-storytelling/blob/master/data_analysis.ipynb)

Insights - 

1. **All albums have exactly 5 corresponding stories**
2. **All stories have exactly 5 sentences**
3. **An image can be part of atmost 5 stories** (meaning it can have 5 annotations)
4. The only attribute in **order** is `story_id`, which can be used throughout the code to make other required fields (listed below), fall in place
5. **All albums have atleast 10 images**
6. **All albums have at-least 1 and at-most 2 image sequences** (these sequences are not necessarily in order)


Other details related to IDs - [PicSOM Instructions doc](https://docs.google.com/document/d/1OWCm0NwKOUq0QETbczHKLRwjFMTO_hEMWEELvysCMmA/edit)

Fields in scope - 

`photo_flickr_id`, `text`, `worker_arranged_photo_order` (just used to order the 5 images in memory at a given time step)

### [PicSOM Instructions file](https://drive.google.com/open?id=18TcQ728-ac1OpWZ5i5Dv_QIoiboR4UlE)