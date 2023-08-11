#for video hong_kong_airport.mp4
#from 8:10 to 9:40 (90s) (hong_kong_airport_demo_data.mp4)
#Resample the video at 5fps, frame index starts from zero. The intervals are an array of tuple of (start,end)
test_cases=[
    #Note: even if there is only one object to be found,
    #'object' should still be a list!
    {
        #general objects
        'object':['white backpack','white suitcase','black backpack','black suitcase'],
        'type':'lang',
    },
    {
        #novel luggage 1 lang
        'object':['black and white striped backpack'],
        'type':'lang',
        'intervals': [('01:10', '01:15')],
    },
    {
        #novel luggage 2 img
        'object':['test_images/pink_short_luggage.jpg'],
        'type':'image',
    },
    {
        #novel luggage 2 lang
        'object':['pink suitcase'],
        'type':'lang',
    },
    {
        #person 1 image
        'object':['test_images/rainbow_shirt_white_shoes_girl.jpg'],
        'type':'image',
    },
    {
        #person 1 lang
        'object':['girl wearing pink coat and black pants with a pink luggage'],
        'type':'lang'
    },
    
    #testcases above already demoed
    #please fine tune these two langauge queries

    {
        #person 2 lang

        'object':['black shirt man'],
        'type':'lang'
    },
    {
        #person 2 img

        'object':['test_images/white_shirt_blue_jeans_man.jpg'],
        'type':'image',
    },
    {
        #person 2 img

        'object':['white shirt blue jeans man'],
        'type':'lang',
    },

    ######## use the testcases above first ########
    {
        'object': ['black backpack'],
        'intervals': [('00:17', '00:19'), ('00:23', '00:24'), ('00:49', '00:50'), ('00:59', '01:14')],
    },
    {
        'object':'white suitcase',
        'intervals': [('00:11', '00:13')],
    },
    {
        'object': 'tote bag',
        'intervals': [('00:17', '00:19'), ('00:31', '00:34'), ('00:55', '00:57'), ('01:37', '01:30')],
    },
    {
        'object': 'pink_short_luggage.jpg',
        'intervals': [('00:17', '00:21')],
    },
    {
        'object': 'silver_luggage.jpg',
        'intervals': [('01:17', '01:25')],
    },
    {
        'object':'white_shirt_blue_jeans_man.jpg',
        'intervals':[('00:07', '00:20')],
    },
    {
        'object': 'suitcase_mask_man.jpg',
        'intervals': [('01:28', '01:32')],
    },
    {
        'object': 'rainbow_shirt_white_shoes_girl.jpg',
        'intervals': [('00:15', '00:21')],
    },
    {
        'object': 'woman_with_phone_trolley.jpg',
        'intervals': [('00:36', '00:44')],
    },

    {
        # negative sample: should not exist in any frame
        'object': 'steak',
        'intervals': [],
    }

    # two more novel luggages (described with color/shape)
    # two mans and two women (one with plain clothing and the other with fancier clothing)
    # images of man and woman and the novel luggages placed in test_images
]