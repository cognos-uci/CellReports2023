response_matching = simple_matching;
default_attenuation = 0;
default_stimulus_time_out = 10;
active_buttons = 8;
button_codes = 32,64,128,128,32,128,128,0; # Mouse click (on), Mouse click (off), right click key, space, Enter, down arrow, up arrow, sound threshold (not used)
pulse_width = 2;  
write_codes = true;      
default_monitor_sounds = false;
default_clear_active_stimuli=true;
response_logging = log_all;

## REMOVE ALL #$$$#

## Some sounds are at max, especially during sleep

/*
4-21: scenes/sounds 1-16 + two unused
96 - space (and "-" in the recall stage)

During wake:
	32 - left mouse click (on) (also enter during the recall part of T2)
	64 - left mouse click (off)
	128 - space
	128 - right click (on)
	33 - ESC
	
	6-26 - scenes by order (26 - practice item scene)
	1-4 - items for each scene
	27 - instruction screen followed immediately (40ms) by:
		1 - Training Wakeup instruction screen (for T2 as well)
		2 - SSS screen (for T2 as well)
		16 - Instructions for localizer task
		3 - "Experimenter instructions" - story building
		4 - Story building "ready to start" screen
		5 - "Experimenter instructions" - spatial task
		6 - "ready for practice" screen
		7 - "Experimenter instructions" - spatial task - phase 2
		8 - "Experimenter instructions" - spatial task - phase 2, with image
		9 - "Refresh your memory for story" screen
		10 - "ready to start placement" screen
		11 - T1 positioning Intro (for T2 as well)
		12 - Thank you (for T2 as well)
		13 - Item-scene recall test instructions
		14 - Call experimenter screen
		15 - Sound heard during sleep intro
		
	28 - "+" screen
	29 - "record your story" screen
	1 - Red Box appears on the left
	2 - Red Box appears on the right
	8 - mouse motion onset
	27 – feedback screen

	Localizer:
	21-23 - scene, object and scrambled images presentation, respectively.
	24 - indication of repeated item.
		
During sleep:
	Sound onset is marked by object number, between 1 and 72 (76 is the novel sound).############ Sound onset is marked by object number, between 1 and 80 (84 is the novel sound).
	85, 86… 95 marks the end of the 1st, 2nd,... 11th full presentation of the whole sound "playlist". 95 will also mark the end of any subsequent presentation (>=11). The timing of this code is arbitrary (~600 ms after sound onset).
	32 - Enter
	128 - Space/Right click (space starts/stops TMR presentation, right click presents the last sound again just once)
	128 (twice) - down arrow
	128 (three times)- up arrow
	
*/



##!# Lines to add/replace line above to check sound volumes during sleep
begin;
	
	text { #used as on screen instruction screen text
		caption = " ";
		formatted_text = true;
		font_color = 0,0,0;
		background_color = 255,255,255;
		transparent_color = 255,255,255;
	} Instruct_text;
	
	picture { #white picture between trials
		background_color = 255,255,255;
		text { #used as on screen instruction screen text
			caption = "<b>+</b>";
			formatted_text = true;
			font_size = 80;
			font_color = 0,0,0;
			background_color = 255,255,255;
			transparent_color = 255,255,255;
		}cursor;	
		x=0;y=0;
	} default;
	
	picture { 
		background_color = 255,255,255;
		default_2d_on_top = true;
		line_graphic {
			line_color = 0,0,0,255;
			fill_color = 255,255,255,255;
			line_width = 2;
		} letter_background;
		x=0;y=0;
		text { 
			caption = "<b>+</b>";
			formatted_text = true;
			font_size = 80;
			font_color = 0,0,0;
			background_color = 255,255,255;
			transparent_color = 255,255,255;
		} nbackletter;	
		x=0;y=0;
		text {
			caption = " ";
			formatted_text = true;
			font_size = 80;
			font_color = 0,0,0;
			background_color = 255,255,255;
			transparent_color = 255,255,255;
		} statustext;
		x=0;y=0;
	} n_back_woscene;

	picture { 
		background_color = 255,255,255;
		default_2d_on_top = true;		
		line_graphic letter_background;
		x=0;y=0;
		text nbackletter;	
		x=0;y=0;
		text statustext;	
		x=0;y=0;
	} n_back_wscene;	

	text { #used as on screen instruction screen text
		caption = "<b>+</b>";
		formatted_text = true;
		font_size = 40;
		font_color = 0,0,0;
		background_color = 255,255,255;
		transparent_color = 255,255,255;
	} MouseMarker;	

	text {
		caption = " ";
		formatted_text = true;
		font_size = 20;
		font_color = 0,0,0;
		background_color = 255,255,255;
		transparent_color = 255,255,255;
	} counter;	

	array {
		LOOP $i 9;
			sound {
				wavefile {
					filename = "\Stim0.wav";
				};
			};
		ENDLOOP;
	} TMRcues;

	sound {
		wavefile {
			filename = "\NapNoise.wav";
		}NoiseNap;
		attenuation = 0.4; 
	} Noise_for_nap;	

	sound {
		wavefile {
			filename = "\Stim0.wav";
		};
		attenuation = 1; 
	} NullSound;	

	picture {} nullpic;

	trial {# used just for the event_code
		trial_duration = 50;
		trial_type = fixed;
		stimulus_event {
			nothing {};
			code = "Instructions";
			port_code = 27;
		};
		stimulus_event {
			nothing {};
			port_code = 0;
			deltat = 40;
		}screentype;
	} NullTrial_wPort;	

	trial {# used just for the event_code
		trial_duration = 20;
		trial_type = fixed;
		stimulus_event {
			nothing {};
			code = "Instructions";
		};
	} NullTrial_woPort;


	trial {# this trial is used to shut down background sounds that are presented but can't be stopped
		trial_duration = 1;
		trial_type = fixed;
		monitor_sounds =true;
		stimulus_event {
			picture nullpic;
		};
	} Silencetrial; 
	

	# this is the square used for the "wake-up" task
	picture {
		background_color = 255,255,255;
		text {
			caption="+";
			font_size=96;
			font_color=0,0,0;
			background_color=255,255,255;
		};
		x=0; y=0;
		box {
			color=255,0,0;
			height=100;
			width=100;
		} redbox;
		x=360; y=0;			
	} redboxpic;

	# this is used for user responses in the "wake-up" task
	trial {
	   trial_duration = 1500;
	   trial_type = specific_response;
		terminator_button=1;
		all_responses = false;
		stimulus_event {
			nothing {};
			duration=response;
			stimulus_time_in=0;
			stimulus_time_out=never;
			target_button=1,2;
			code="Wake-up - trial stop";
		}wakeuptaskStim;
	} wakeuptask;
	
	trial {
		video {
			filename = "//zoom_0.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid0;
		time = 0;
	}vidpresent0;

	trial {
		video {
			filename = "//zoom_1.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid1;
		time = 0;
	}vidpresent1;
	
	trial {
		video {
			filename = "//zoom_2.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid2;
		time = 0;
	}vidpresent2;

	trial {
		video {
			filename = "//zoom_3.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid3;
		time = 0;
	}vidpresent3;

	trial {
		video {
			filename = "//zoom_4.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid4;
		time = 0;
	}vidpresent4;
	
	trial {
		video {
			filename = "//zoom_5.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid5;
		time = 0;
	}vidpresent5;
		
	trial {
		video {
			filename = "//zoom_6.avi";
			x = 0; y = 0;
			height = 810; width = 1440;
		}vid6;
		time = 0;
	}vidpresent6;
	
begin_pcl;

	string SubjectNum = logfile.subject();
	string RunType=parameter_manager.get_string ("Run Type");#Training, Sleep or T2
	int Radius=display_device.height()/2; # Outer radius of the display circle, in pixels
	int inner_clear_Radius=50;# Outer radius of the display circle
	int margin=50; #number of pixels on the outer rim of the circle in which no images will be positioned
	int num_rings=8; # number of rings in the checkerboard
	int num_slices=8;# number of triangular slices in the checkerboard
	int correct_incorrect_criterion=100;#pixels
	double scaling_factor=double(display_device.width())/1920; # the task was designed for a 1920 pixel wide screen, so size have to scaled accordingly
	Instruct_text.set_font_size(int(40*scaling_factor));
	int pic_side_size=125;
	int scene_width=652;int scene_height=326;
	inner_clear_Radius=int(double(inner_clear_Radius)*scaling_factor);
	margin=int(double(margin)*scaling_factor);
	correct_incorrect_criterion=int(double(correct_incorrect_criterion)*scaling_factor);
	int num_scenes=18;#20
	int num_images_per_scene=4;
	pic_side_size=int(double(pic_side_size)*scaling_factor);
	scene_width=int(double(scene_width)*scaling_factor);
	scene_height=int(double(scene_height)*scaling_factor);
	int box_size=300; # size of boxes for new/old test and color test
	box_size=int(box_size*scaling_factor);
	array<double>box_square[4][2];
	array<double>unit_square[][]={{-1.0,-1},{-1,1},{1,1},{1,-1}}; #Used to set the frame square around the image
	loop int i=1 until i>4 begin loop int j=1 until j>2 begin box_square[i][j]=unit_square[i][j]*box_size/2; j=j+1; end; i=i+1; end;
	#array<string>Questions[]={"Did the object appear at\nthe very beginning of the story?","Did the object appear at\nthe very end of the story?","Did the object appear throughout\nthe whole story, start to end?","Did any person see the object\nas part of the story?","Was the object in motion\n(not static) during the story?","Did the object produce a\nsound as part of the story?"};#"Was this object held\nin someone's hand?"
	array<string>Questions[]={"Did the object appear throughout\nthe whole story, start to end?","Was the object in motion\n(not static) during the story?"};#"Was this object held\nin someone's hand?"
	array<int>Answers[num_scenes][Questions.count()][num_images_per_scene];
	int y_shift_left=-100;
	int num_practice_images=3;
	int YNDelay=2000;#ms
	array<string>objectnames[]={"Volleyball","Lighter","Blue-Jay","Book","Camera","Car","Cat","Clock","Zipper","Dog","Toilet","Juice","Apple","Frog","Heart","Kettle","Wand","Train","Phone","Piano","Plane","Duck","Rose","Plate","Saw","Violin","Lion","Monkey","Pen","Lips","Cow","Pig","Record","Register","Door-bell","Toothbrush","Robot","Owl","Gong","Call-bell","Faucet","Pot","Spring","Balloon","Keyboard","Mouthwash","Wine","Hands","Can","Fly","Recorder","Keys","Helicopter","Motorcycle","Bowling-Pin","Crow","Elephant","Drum","Donkey","Sword","Harmonica","Rooster","Saxophone","Sheep","Vacuum","Whistle","Guitar","Fire-engine","Cord","Cuckoo-Clock","Baby","Cricket","Canon","Bagpipes","Hammer","Scissors"};
	
## Graphics and trial initialization
###################################################################################################################################################################################################

	int time_limit=120;#seconds
	array<double> item_frame_dimensions[4][2];
	array<double> item_frame_dimensions_left[4][2];
	array<double> item_frame_dimensions_large[4][2];
	line_graphic item_frame=new line_graphic;
	line_graphic item_frame_large=new line_graphic;
	line_graphic item_frame_left=new line_graphic;
	#item_frame.set_line_color(colors[circ_color_num][1],colors[circ_color_num][2],colors[circ_color_num][3],255);
	item_frame.set_fill_color(255,255,255,0);		item_frame_large.set_fill_color(255,255,255,0);		item_frame_left.set_fill_color(255,255,255,0);
	 double left_line_width=2;
	item_frame.set_line_width(4);		item_frame_large.set_line_width(14);		item_frame_left.set_line_width(left_line_width);
	loop int i=1 until i>4 begin loop int j=1 until j>2 begin item_frame_dimensions[i][j]=unit_square[i][j]*pic_side_size/2; item_frame_dimensions_large[i][j]=3*unit_square[i][j]*pic_side_size/2; j=j+1; end; i=i+1; end;
	#loop int i=1 until i>4 begin  item_frame_dimensions_left[i][1]=unit_square[i][1]*(double(pic_side_size)/4)/2; item_frame_dimensions_left[i][2]=unit_square[i][2]*(double(pic_side_size)/10)/2; i=i+1; end;
	item_frame_dimensions_left[1][1]=unit_square[1][1]*(double(pic_side_size)/4)/2-left_line_width/2; item_frame_dimensions_left[1][2]=unit_square[1][2]*(double(pic_side_size)/10)/2-left_line_width/2;
	item_frame_dimensions_left[2][1]=unit_square[2][1]*(double(pic_side_size)/4)/2-left_line_width/2; item_frame_dimensions_left[2][2]=unit_square[2][2]*(double(pic_side_size)/10)/2+left_line_width/2;
	item_frame_dimensions_left[3][1]=unit_square[3][1]*(double(pic_side_size)/4)/2+left_line_width/2; item_frame_dimensions_left[3][2]=unit_square[3][2]*(double(pic_side_size)/10)/2+left_line_width/2;
	item_frame_dimensions_left[4][1]=unit_square[4][1]*(double(pic_side_size)/4)/2+left_line_width/2; item_frame_dimensions_left[4][2]=unit_square[4][2]*(double(pic_side_size)/10)/2-left_line_width/2;
	
	item_frame.add_polygon(item_frame_dimensions,true,1,0);		item_frame_large.add_polygon(item_frame_dimensions_large,true,1,0);		item_frame_left.add_polygon(item_frame_dimensions_left,true,1,0);
	item_frame.redraw();	item_frame_large.redraw();	item_frame_left.redraw();

	# Basic picture initialization
	#OLD#array<int>colors[][]={{128,0,0},{170,110,40},{255,255,255},{79,165,180},{52,86,67},{146,159,176},{186,117,48},{126,78,180},{0,255,0},{93,52,32},{241,236,18},{139,170,108},{235,209,186},{200,200,200},{217,173,83},{240,50,230},{230,25,75},{205,107,18},{0,130,200},{0,0,128},{170,255,195}};
	array<int>colors[num_scenes+1][3];
	if num_scenes==18 then
		colors={{128,0,0},{170,110,40},{255,255,255},{79,165,180},{52,86,67},{146,159,176},{186,117,48},{126,78,180},{0,255,0},{93,52,32},{241,236,18},{139,170,108},{0,0,128},{200,200,200},{217,173,83},{240,50,230},{230,25,75},{0,130,200},{170,255,195}};
	elseif num_scenes==20 then
		colors={{128,0,0},{170,110,40},{255,255,255},{79,165,180},{52,86,67},{146,159,176},{186,117,48},{126,78,180},{0,255,0},{93,52,32},{241,236,18},{139,170,108},{0,0,128},{200,200,200},{217,173,83},{240,50,230},{230,25,75},{0,130,200},{170,255,195},{205,107,18},{235,209,186}};		
	end;	
	picture emptypic=new picture; 
	emptypic.set_background_color(255,255,255);
	array<picture> circle_pic[colors.count()];
	array<picture> no_circle_pic[colors.count()];
	text status_txt=new text;
	status_txt.set_caption(" ");
	status_txt.set_font_size(int(30*scaling_factor));
	status_txt.set_background_color(255,255,255,0);
	status_txt.set_font_color(0,0,0);
	status_txt.redraw();
	bitmap roomimage=new bitmap("\Scenes\\Room.jpg");roomimage.set_load_size(900,777,0);roomimage.load();
	bitmap Background=new bitmap("\Scenes\\Background.jpg");Background.set_load_size(display_device.height()*1.1,display_device.height()*1.1,0);Background.load();
	array<bitmap> scenes[num_scenes+1];
	
	int num_scenes_per_block=2;#############
	#int num_scenes_per_block=4;############
	array<int>scenes_per_block[num_scenes];
	scenes_per_block.fill(1,scenes_per_block.count(),1,1);
	scenes_per_block.shuffle();	
	
	loop int circ_color_num=1 until circ_color_num>colors.count() begin
		circle_pic[circ_color_num]=new picture;
		circle_pic[circ_color_num].set_default_2d_on_top(false);
		circle_pic[circ_color_num].set_background_color(255,255,255);

		no_circle_pic[circ_color_num]=new picture;
		no_circle_pic[circ_color_num].set_default_2d_on_top(false);
		no_circle_pic[circ_color_num].set_background_color(255,255,255);

		cuboid big_circle_graphic=new cuboid(double(display_device.width()),double(display_device.height()),0.0);
		bitmap small_circle_graphic=new bitmap("\Circles\\C"+string(circ_color_num)+".tif");small_circle_graphic.set_load_size(display_device.height()/10,display_device.width()/4,0);small_circle_graphic.set_transparent_color(250,250,250);small_circle_graphic.load();
		texture textr=new texture();
		textr.set_filename("\Circles\\C"+string(circ_color_num)+".tif");	textr.load();
		textr.set_transparent_color(250,250,250);
		big_circle_graphic.set_texture(textr);
		big_circle_graphic.set_emissive(1.0,1.0,1.0);		
		big_circle_graphic.rotate_mesh(0,0.0,0.0);
		#if circ_color_num<=num_scenes then
			scenes[circ_color_num]=new bitmap("\Scenes\\C"+string(circ_color_num)+".jpg");
			scenes[circ_color_num].set_load_size(scene_height,scene_width,0);scenes[circ_color_num].load();
			array<double> frame_dimensions[4][2];
			line_graphic frame=new line_graphic;
			frame.set_line_color(colors[circ_color_num][1],colors[circ_color_num][2],colors[circ_color_num][3],255);
			frame.set_fill_color(255,255,255,0);
			frame.set_line_width(5);
			loop int i=1 until i>4 begin frame_dimensions[i][1]=unit_square[i][1]*scene_width/2; frame_dimensions[i][2]=unit_square[i][2]*scene_height/2; i=i+1; end;
			frame.add_polygon(frame_dimensions,true,1,0);
			frame.redraw();
			circle_pic[circ_color_num].add_part(Background,display_device.width()/2-Radius,0);
			circle_pic[circ_color_num].add_part(roomimage,-Radius-25,y_shift_left);
			circle_pic[circ_color_num].add_part(scenes[circ_color_num],-Radius-25,250+y_shift_left);
			circle_pic[circ_color_num].add_part(frame,-Radius-25,250+y_shift_left);
			circle_pic[circ_color_num].add_part(small_circle_graphic,-565,-250+y_shift_left);	
			no_circle_pic[circ_color_num].add_part(roomimage,-Radius-25,y_shift_left);
			no_circle_pic[circ_color_num].add_part(scenes[circ_color_num],-Radius-25,250+y_shift_left);
			no_circle_pic[circ_color_num].add_part(frame,-Radius-25,250+y_shift_left);
		#end;
		circle_pic[circ_color_num].add_3dpart(big_circle_graphic,double(display_device.width()/2-Radius),0.0,0.0);	
		circle_pic[circ_color_num].add_part(status_txt,display_device.width()/2-100,-display_device.height()/2+100);
		#circle_pic[circ_color_num].present();
		#wait_interval(3000);
		circ_color_num=circ_color_num+1;
	end;
	
	statustext.set_font_size(int(30*scaling_factor));	statustext.redraw();

	# Basic trial initialization
	int sound_picture_delay=1500;#ms
	int picture_exposure_time=2000;#ms
	int post_picture_delay=1000;#ms
	int exposure_ITI=1000;#ms
	int exposure_duration_in_ms=sound_picture_delay+picture_exposure_time+post_picture_delay;
	trial exposure_trial=new trial;
	exposure_trial.set_duration(exposure_duration_in_ms);
	stimulus_event ImageStim_for_exposure1=exposure_trial.add_stimulus_event(new nothing);
	ImageStim_for_exposure1.set_time(0);
	ImageStim_for_exposure1.set_event_code("Exposure Trial - Checkerboard + object (1)");
	stimulus_event SoundStim_for_exposure1=exposure_trial.add_stimulus_event(new nothing);
	SoundStim_for_exposure1.set_delta_time(0);
	stimulus_event ImageStim_for_exposure2=exposure_trial.add_stimulus_event(new nothing);
	ImageStim_for_exposure2.set_delta_time(sound_picture_delay);
	ImageStim_for_exposure2.set_event_code("Exposure Trial - Checkerboard + object + scene");
	stimulus_event ImageStim_for_exposure3=exposure_trial.add_stimulus_event(new nothing);
	ImageStim_for_exposure3.set_delta_time(picture_exposure_time);
	ImageStim_for_exposure3.set_event_code("Exposure Trial - Checkerboard + object (2)");
	stimulus_event SoundStim_for_exposure2=exposure_trial.add_stimulus_event(new nothing);
	SoundStim_for_exposure2.set_delta_time(0);

	int sound_object_delay_for_pos=1000;#ms
	int positioning_ITI=1000;#ms
	int feedback_duration_in_ms=2000;#ms
	/*trial positioning_trial=new trial;
	positioning_trial.set_duration(sound_object_delay_for_pos);
	stimulus_event ImageStim_for_positioning=positioning_trial.add_stimulus_event(circle_pic[place_holder]);
	ImageStim_for_positioning.set_time(0);
	ImageStim_for_positioning.set_event_code("Positioning Trial - Circle + Object Presentation");
	stimulus_event SoundStim_for_positioning=positioning_trial.add_stimulus_event(new nothing);
	SoundStim_for_positioning.set_delta_time(0);
	stimulus_event Nothing_for_positioning=positioning_trial.add_stimulus_event(new nothing);#just for code
	Nothing_for_positioning.set_delta_time(100);
	#Nothing_for_positioning.set_duration(sound_object_delay_for_pos-100);
	*/
	
	# Mouse initialization
	mouse mse = response_manager.get_mouse(1);
	mse.set_min_max( 1, -display_device.width() / 2, display_device.width() / 2);
	mse.set_min_max( 2, -display_device.height() / 2, display_device.height() / 2 );
	mse.set_min_max(3,-960,960);
	mse.set_restricted( 1, true );
	mse.set_restricted( 2, true );

	# Instruction screen without sound example initialization
	picture InstructPicWithoutImage=new picture;
	InstructPicWithoutImage.set_background_color(255,255,255);
	InstructPicWithoutImage.add_part(Instruct_text,0,0);

	picture InstructPicWithImage=new picture;
	InstructPicWithImage.set_background_color(255,255,255);
	InstructPicWithImage.add_part(Instruct_text,0,0);
	bitmap bullseye=new bitmap("Bullseye.tif");bullseye.set_load_size(0,0,0.5);bullseye.load();
	InstructPicWithImage.add_part(bullseye,0,-100);
	
	wavefile tempsoundforfilename=new wavefile("./1.wav");
	string Just_for_the_dir_name=tempsoundforfilename.filename();
	Just_for_the_dir_name=Just_for_the_dir_name.replace("stim\\1.wav","log\\S"+SubjectNum+"-"+RunType+".txt");
	loop int i=1 until !file_exists(Just_for_the_dir_name) begin
		if i==1 then
			Just_for_the_dir_name=Just_for_the_dir_name.replace(".txt","_"+string(i)+".txt");
		else
			Just_for_the_dir_name=Just_for_the_dir_name.replace("_"+string(i-1)+".txt","_"+string(i)+".txt");
		end;
		i=i+1;
	end;
	output_file out_results = new output_file;	
	
	array<int>item_scene_allocation[num_scenes*num_images_per_scene+4]; # the "+4" is for and three practice items and an item presented only during sleep
	array<int>true_positions[num_scenes*num_images_per_scene][2];
	
	bool FileOpen;
	int Initial_max_response_time;
	double sum_bonus=0;
	output_file out = new output_file;
	if file_exists(tempsoundforfilename.filename().replace("stim\\1.wav","log\\Param_"+SubjectNum+".txt")) then # If the subject was run before on any condition, the stimuli was saved and is now loaded
		term.print_line("Loading parameter file: "+tempsoundforfilename.filename().replace("stim\\1.wav","log\\Param_"+SubjectNum+".txt"));
		input_file in = new input_file;
		in.open(".\\log\\Param_"+SubjectNum+".txt");
		in.set_delimiter( '\n' );
		loop int i=1 until i>item_scene_allocation.count() begin
			item_scene_allocation[i]=in.get_int();
			i=i+1;
		end;
		loop int i=1 until i>num_scenes*num_images_per_scene begin
			true_positions[i][1]=in.get_int();
			true_positions[i][2]=in.get_int();
			true_positions[i][1]=true_positions[i][1]+display_device.width()/2-Radius;
			i=i+1;
		end;
		loop int i=1 until i>num_scenes begin
			scenes_per_block[i]=in.get_int();
			i=i+1;
		end;
		Initial_max_response_time=in.get_int();
		sum_bonus=in.get_double();
		in.close();	
		term.print_line("Previous bonus sum is: "+string(sum_bonus)+"$");
	else		
		term.print_line("Creating parameter file: "+tempsoundforfilename.filename().replace("stim\\1.wav","log\\Param_"+SubjectNum+".txt"));		
		out.open(".\\Param_"+SubjectNum+".txt");
		item_scene_allocation.fill(1,item_scene_allocation.count(),1,1);
		item_scene_allocation.shuffle();		
		loop int i=1 until i>item_scene_allocation.count() begin
			out.print(item_scene_allocation[i]);out.print( "\n" );
			i=i+1;
		end;
		
		input_file incoords = new input_file;
		incoords.open(".\\stim\\coords\\coord"+SubjectNum+".txt");
		incoords.set_delimiter( '\n' );
		loop int i=1 until i>num_scenes*num_images_per_scene begin
			true_positions[i][1]=incoords.get_int();
			true_positions[i][2]=incoords.get_int();
			out.print(true_positions[i][1]);out.print( "\n" );
			out.print(true_positions[i][2]);out.print( "\n" );
			true_positions[i][1]=true_positions[i][1]+display_device.width()/2-Radius;
			i=i+1;
		end;
		loop int i=1 until i>num_scenes begin
		out.print(scenes_per_block[i]);out.print( "\n" );			
			i=i+1;
		end;
		
		FileOpen=true;
		#out.close();
	end;
	
	if file_exists(tempsoundforfilename.filename().replace("stim\\1.wav","log\\Answers_"+SubjectNum+".txt")) then
		input_file inAnswers = new input_file;
		inAnswers.open(".\\log\\Answers_"+SubjectNum+".txt");
		inAnswers.set_delimiter( '\n' );
		loop int i=1 until i>num_scenes begin
			loop int j=1 until j>Questions.count() begin
				loop int n=1 until n>num_images_per_scene begin
					Answers[i][j][n]=inAnswers.get_int();
					n=n+1;
				end;
				j=j+1;
			end;
			i=i+1;
		end;
		inAnswers.close();
	end;
	
	if RunType=="T2" || RunType=="Training" then
		string filename=Just_for_the_dir_name;
		loop int i=1 until !file_exists(filename) begin		if i==1 then 			filename=filename.replace(".txt","_"+string(i)+".txt");		else			filename=filename.replace("_"+string(i-1)+".txt","_"+string(i)+".txt");		end;		i=i+1;	end;
		out_results.open(filename);
		term.print("Saving in file: "+filename+"; ");
		out_results.print("Stage\tBlock\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename\n");
	end;	
	array<double>atten_per_wav_file[item_scene_allocation.count()];	
	if RunType=="Sleep" then
		#OLD#atten_per_wav_file={0.17,0.21,0.3,0.21,0.25,0.29,0.28,0.22,0.29,0.25,0.22,0.21,0.19,0.23,0.17,0.24,0.29,0.26,0.28,0.29,0.31,0.31,0.22,0.18,0.21,0.3,0.25,0.26,0.27,0.24,0.23,0.31,0.32,0.28,0.24,0.27,0.26,0.27,0.21,0.26,0.15,0.22,0.17,0.26,0.18,0,0.16,0.21,0.14,0.21,0.15,0.24,0.16,0.22,0.24,0.2,0.23,0.21,0.22,0.21,0.27,0.24,0.24,0.23,0.17,0.29,0.2,0.17,0.22,0.25,0.18,0,0.17,0.23,0,0.02,0,0.04,0,0,0,0.03,0.04,0.02};
		if num_scenes==18 then
			atten_per_wav_file={0.28,0.32,0.39,0.29,0.35,0.38,0.38,0.31,0.37,0.38,0.26,0.26,0.25,0.34,0.23,0.33,0.38,0.35,0.36,0.39,0.4,0.33,0.34,0.31,0.29,0.41,0.34,0.34,0.37,0.33,0.31,0.39,0.39,0.37,0.14,0.34,0.33,0.38,0.36,0.37,0.24,0.29,0.28,0.34,0.25,0.28,0.26,0.29,0.23,0.29,0.21,0.3,0.29,0.26,0.31,0.29,0.31,0.27,0.28,0.27,0.31,0.32,0.3,0.31,0.26,0.34,0.31,0.38,0.27,0.33,0.3,0.35,0.28,0.29,0.31,0.3};
		elseif num_scenes==20 then
			atten_per_wav_file={0.28,0.32,0.39,0.29,0.35,0.38,0.38,0.31,0.37,0.38,0.26,0.26,0.25,0.34,0.23,0.33,0.38,0.35,0.36,0.39,0.4,0.33,0.34,0.31,0.29,0.41,0.34,0.34,0.37,0.33,0.31,0.39,0.39,0.37,0.14,0.34,0.33,0.38,0.36,0.37,0.24,0.29,0.28,0.34,0.25,0.28,0.26,0.29,0.23,0.29,0.21,0.3,0.29,0.26,0.31,0.29,0.31,0.27,0.28,0.27,0.31,0.32,0.3,0.31,0.26,0.34,0.31,0.38,0.27,0.33,0.3,0.35,0.28,0.29,0.31,0.3,0.4,0.34,0.08,0.24,0.25,0.1,0.13,0.27};
		end;

		#atten_per_wav_file.fill(1,atten_per_wav_file.count(),0.3,0);
	else
		#OLD#atten_per_wav_file={0.18,0.24,0.27,0.23,0.25,0.29,0.25,0.25,0.28,0.26,0.24,0.23,0.22,0.24,0.18,0.23,0.28,0.26,0.28,0.26,0.26,0.31,0.24,0.24,0.23,0.26,0.27,0.26,0.29,0.24,0.24,0.32,0.3,0.26,0.23,0.24,0.27,0.25,0.22,0.24,0.18,0.23,0.2,0.28,0.19,0.12,0.18,0.24,0.16,0.25,0.15,0.25,0.22,0.26,0.26,0.2,0.23,0.24,0.19,0.25,0.24,0.22,0.23,0.23,0.19,0.27,0.23,0.18,0.25,0.25,0.21,0.05,0.21,0.25,0.03,0.07,0,0.06,0.04,0,0,0.09,0.11,0.05};
		#atten_per_wav_file={0.18,0.24,0.27,0.23,0.25,0.29,0.25,0.25,0.28,0.26,0.24,0.23,0.22,0.24,0.18,0.23,0.28,0.26,0.28,0.26,0.26,0,0.24,0.24,0.23,0.26,0.27,0.26,0.29,0.24,0.24,0.32,0.3,0.26,0.06,0.24,0.27,0.25,0.22,0.24,0.18,0.23,0.2,0.28,0.19,0.04,0.18,0.24,0.16,0.25,0.15,0.25,0,0.26,0.26,0.2,0.23,0.24,0.19,0.25,0.24,0.22,0.23,0.23,0.19,0.27,0.23,0,0.25,0.25,0.05,0.05,0.21,0.25,0.03,0.07,0.31,0.23,0.12,0.22,0.18,0.09,0.11,0.21};
		if num_scenes==18 then
			atten_per_wav_file={0.18,0.24,0.27,0.23,0.25,0.29,0.25,0.25,0.28,0.26,0.24,0.23,0.22,0.24,0.18,0.23,0.28,0.26,0.28,0.26,0.26,0.25,0.24,0.22,0.23,0.26,0.27,0.26,0.29,0.24,0.24,0.32,0.3,0.26,0.06,0.24,0.27,0.25,0.22,0.24,0.18,0.23,0.2,0.28,0.19,0.24,0.18,0.24,0.16,0.25,0.15,0.25,0.25,0.26,0.26,0.2,0.23,0.24,0.19,0.25,0.24,0.22,0.23,0.23,0.19,0.27,0.23,0.35,0.25,0.25,0.25,0.35,0.21,0.25,0.23,0.27};
		elseif num_scenes==20 then
			atten_per_wav_file={0.18,0.24,0.27,0.23,0.25,0.29,0.25,0.25,0.28,0.26,0.24,0.23,0.22,0.24,0.18,0.23,0.28,0.26,0.28,0.26,0.26,0.25,0.24,0.22,0.23,0.26,0.27,0.26,0.29,0.24,0.24,0.32,0.3,0.26,0.06,0.24,0.27,0.25,0.22,0.24,0.18,0.23,0.2,0.28,0.19,0.24,0.18,0.24,0.16,0.25,0.15,0.25,0.25,0.26,0.26,0.2,0.23,0.24,0.19,0.25,0.24,0.22,0.23,0.23,0.19,0.27,0.23,0.35,0.25,0.25,0.25,0.35,0.21,0.25,0.23,0.27,0.31,0.23,0.12,0.22,0.18,0.09,0.11,0.21};
		end;
		#atten_per_wav_file.fill(1,atten_per_wav_file.count(),0.2,0);
	end;
	
	array<cuboid>objects[item_scene_allocation.count()];
	array<texture>textures_for_objects[item_scene_allocation.count()];
	array<sound>sounds[item_scene_allocation.count()];
		
	loop int i=1 until i>item_scene_allocation.count() begin
		Just_for_the_dir_name=tempsoundforfilename.filename();
		sounds[i]=new sound(new wavefile(Just_for_the_dir_name.replace("1",string(i))));
		sounds[i].set_attenuation(atten_per_wav_file[i]);
		i=i+1;
	end;
	Just_for_the_dir_name=tempsoundforfilename.filename();
	loop int i=1 until i>item_scene_allocation.count() begin
		textures_for_objects[i]=new texture();
		textures_for_objects[i].set_filename(Just_for_the_dir_name.replace("1.wav",string(i)+".jpg"));
		textures_for_objects[i].load();
		objects[i]=new cuboid(double(pic_side_size), double(pic_side_size), 0.0);
		objects[i].set_emissive(1.0,1.0,1.0);		
		objects[i].set_texture(textures_for_objects[i]);
		textures_for_objects[i].set_alpha(255);
		i=i+1;
	end;	
	cuboid img=new cuboid(double(pic_side_size),double(pic_side_size),0.0);img.set_emissive(1.0,1.0,1.0);
	bitmap img_left=new bitmap; img_left.set_load_size(double(pic_side_size)/10,double(pic_side_size)/4,0);
	
	true_positions.resize(true_positions.count()+3);#practice items
	true_positions[true_positions.count()-2][1]=Radius/2+display_device.width()/2-Radius;
	true_positions[true_positions.count()-2][2]=Radius/2;
	true_positions[true_positions.count()-1][1]=-Radius/2+display_device.width()/2-Radius;
	true_positions[true_positions.count()-1][2]=0;
	true_positions[true_positions.count()][1]=Radius/2+display_device.width()/2-Radius;
	true_positions[true_positions.count()][2]=-Radius/2;
	

## Localizer initialization
###################################################################################################################################################################################################

	int num_images_for_localizer=40;
	array<bitmap>loc_pic[num_images_for_localizer*3];
	int loc_present_time=1000;#ms
	int loc_ITI_min=2500;#ms
	int loc_ITI_max=3500;#ms
	#int loc_within_stim_break_min=2;
	#int loc_within_stim_break_max=18;
	int loc_within_stim_break_min=1;
	int loc_within_stim_break_max=1;
	double loc_proportion_reps=0.25;

	loop int ii=1 until ii>num_images_for_localizer begin
		loc_pic[ii]=new bitmap("\Localizer\\A"+string(ii)+".jpg");	loc_pic[ii].set_load_size(450,450,0);	loc_pic[ii].load();
		loc_pic[num_images_for_localizer+ii]=new bitmap("\Localizer\\B"+string(ii)+".jpg");	loc_pic[num_images_for_localizer+ii].set_load_size(450,450,0);	loc_pic[num_images_for_localizer+ii].load();
		loc_pic[num_images_for_localizer*2+ii]=new bitmap("\Localizer\\C"+string(ii)+".jpg");	loc_pic[num_images_for_localizer*2+ii].set_load_size(450,450,0);	loc_pic[num_images_for_localizer*2+ii].load();
		ii=ii+1;
	end;
	picture loc_ITI=new picture;
	loc_ITI.set_background_color(193,193,193);
	text ITI_cross=new text;
	ITI_cross.set_caption("<b>+</b>");
	ITI_cross.set_formatted_text(true);	
	ITI_cross.set_font_size(96);
	ITI_cross.set_background_color(193,193,193,0);
	ITI_cross.set_font_color(0,0,0);
	ITI_cross.redraw();
	loc_ITI.add_part(ITI_cross,0,0);

	array<int>tmp2[num_images_for_localizer*3];
	tmp2.fill(1,tmp2.count(),1,1);
	tmp2.shuffle(1,num_images_for_localizer);tmp2.shuffle(num_images_for_localizer+1,2*num_images_for_localizer);tmp2.shuffle(2*num_images_for_localizer+1,3*num_images_for_localizer);
	array<int>loc_rep_items[0];
	loop int i=1 until i>int(num_images_for_localizer*loc_proportion_reps) begin
		loc_rep_items.add(tmp2[i]);loc_rep_items.add(tmp2[num_images_for_localizer+i]);loc_rep_items.add(tmp2[2*num_images_for_localizer+i]);
		i=i+1;
	end;
	loc_rep_items.shuffle();
	loop int i=int(num_images_for_localizer*loc_proportion_reps)+1 until i>num_images_for_localizer begin
		loc_rep_items.add(tmp2[i]);loc_rep_items.add(tmp2[num_images_for_localizer+i]);loc_rep_items.add(tmp2[2*num_images_for_localizer+i]);
		i=i+1;
	end;
	loc_rep_items.shuffle(int(num_images_for_localizer*3*loc_proportion_reps)+1,loc_rep_items.count());

	array<int>loc_dist_reps[int(num_images_for_localizer*3*loc_proportion_reps)];
	array<int>loc_1st_pos_reps[int(num_images_for_localizer*3*loc_proportion_reps)];
	array<int>loc_stims[int(num_images_for_localizer*3*(1+loc_proportion_reps))][3];
	loop int i=1 until i>loc_dist_reps.count() begin
		loc_dist_reps[i]=random(loc_within_stim_break_min,loc_within_stim_break_max);	
		loop bool cond=false; int loc_counter=1 until cond==true begin
			if loc_counter>100 then
				i=1;
				loc_1st_pos_reps.resize(0);loc_1st_pos_reps.resize(int(num_images_for_localizer*3*loc_proportion_reps));
				loc_dist_reps.resize(0);loc_dist_reps.resize(int(num_images_for_localizer*3*loc_proportion_reps));
				loop int j=1 until j>loc_stims.count() begin loc_stims[j][1]=0;loc_stims[j][2]=0;loc_stims[j][3]=0;j=j+1;end;
				loc_counter=1;	
				loc_dist_reps[i]=random(loc_within_stim_break_min,loc_within_stim_break_max);
			end;				
			loc_1st_pos_reps[i]=random(1,int(num_images_for_localizer*3*(1+loc_proportion_reps)));
			cond=true;
			if loc_1st_pos_reps[i]+loc_dist_reps[i]>int(num_images_for_localizer*3*(1+loc_proportion_reps)) then
				cond=false;
				loc_counter=loc_counter+1;
				continue;
			end;
			loop int j=1 until j==i begin
				if loc_1st_pos_reps[i]==loc_1st_pos_reps[j] || loc_1st_pos_reps[i]==loc_1st_pos_reps[j]+loc_dist_reps[j] || loc_1st_pos_reps[j]==loc_1st_pos_reps[i]+loc_dist_reps[i] || loc_1st_pos_reps[j]+loc_dist_reps[j]==loc_1st_pos_reps[i]+loc_dist_reps[i] then
					cond=false;
					loc_counter=loc_counter+1;
					break;
				end;
				j=j+1;
			end;
		end;
		loc_stims[loc_1st_pos_reps[i]][1]=loc_rep_items[i];
		loc_stims[loc_1st_pos_reps[i]+loc_dist_reps[i]][1]=loc_rep_items[i];
		loc_stims[loc_1st_pos_reps[i]+loc_dist_reps[i]][3]=1;
		i=i+1;
	end;
	loop int i=1; int j=1 until i>loc_stims.count() begin
		if loc_stims[i][1]==0 then
			loc_stims[i][1]=loc_rep_items[loc_dist_reps.count()+j];
			j=j+1;
		end;
		loc_stims[i][2]=random(loc_ITI_min,loc_ITI_max);
		i=i+1;
	end;
	trial loc_trial = new trial();
	loc_trial.set_type(specific_response);
	loc_trial.set_terminator_button(4);
	loc_trial.set_duration(stimuli_length);
	stimulus_event loc_stim_ev=loc_trial.add_stimulus_event(loc_ITI);
	loc_stim_ev.set_event_code("Localization Start");
	loc_stim_ev.set_port_code(28);
	loc_stim_ev.set_duration(loc_ITI_max);
	int loc_duration=loc_ITI_max;
	loop int i=1 until i>loc_stims.count() begin
		picture loc_pic_screen=new picture;
		loc_pic_screen.set_background_color(193,193,193);
		loc_pic_screen.add_part(loc_pic[loc_stims[i][1]],0,0);
		loc_stim_ev=loc_trial.add_stimulus_event(loc_pic_screen);
		loc_stim_ev.set_delta_time(loc_trial.get_stimulus_event(loc_trial.stimulus_event_count()-1).duration());
		if ((loc_stims[i][1]-1)/num_images_for_localizer)==0 then
			loc_stim_ev.set_event_code("Scene , trial #"+string(i)+", pic A"+string(mod(loc_stims[i][1]-1,num_images_for_localizer)+1));
			loc_stim_ev.set_port_code(21);
		elseif ((loc_stims[i][1]-1)/num_images_for_localizer)==1 then
			loc_stim_ev.set_event_code("Object, trial #"+string(i)+", pic B"+string(mod(loc_stims[i][1]-1,num_images_for_localizer)+1));
			loc_stim_ev.set_port_code(22);
		elseif ((loc_stims[i][1]-1)/num_images_for_localizer)==2 then
			loc_stim_ev.set_event_code("Scramb, trial #"+string(i)+", pic C"+string(mod(loc_stims[i][1]-1,num_images_for_localizer)+1));
			loc_stim_ev.set_port_code(23);
		end;
		if loc_stims[i][3]==1 then
			loc_stim_ev.set_duration(100);
			loc_stim_ev=loc_trial.add_stimulus_event(loc_pic_screen);
			loc_stim_ev.set_delta_time(loc_trial.get_stimulus_event(loc_trial.stimulus_event_count()-1).duration());
			loc_stim_ev.set_port_code(24);	
			loc_stim_ev.set_duration(loc_present_time-100);
		else
			loc_stim_ev.set_duration(loc_present_time);
		end;
		loc_duration=loc_duration+loc_present_time;
		loc_stim_ev=loc_trial.add_stimulus_event(loc_ITI);
		loc_stim_ev.set_delta_time(loc_trial.get_stimulus_event(loc_trial.stimulus_event_count()-1).duration());
		loc_stim_ev.set_event_code("Localization ITI");
		loc_stim_ev.set_port_code(28);
		loc_stim_ev.set_duration(loc_stims[i][2]);
		loc_duration=loc_duration+loc_stims[i][2];
		i=i+1;
	end;
	loc_trial.set_duration(loc_duration);
	#term.print_line(loc_stims);
	#term.print_line(loc_1st_pos_reps);term.print_line(arithmetic_mean(loc_1st_pos_reps));term.print_line(arithmetic_mean(loc_dist_reps));term.print_line((loc_duration-loc_present_time*int(double(num_images_for_localizer)*3*(1.0+loc_proportion_reps))-loc_ITI_max)/int(double(num_images_for_localizer)*3*(1.0+loc_proportion_reps)));

#################################################################################################################################################################
######################################### SUB ROUTINES ##########################################################################################################
#################################################################################################################################################################

	sub bool instructions (string intsruction_text, int screennum) begin
		string temptext=Instruct_text.caption();
		Instruct_text.set_caption(intsruction_text);Instruct_text.redraw();
		intsruction_text.resize(100);
		NullTrial_wPort.get_stimulus_event(1).set_event_code(intsruction_text);
		array<double> tempport[1];NullTrial_wPort.get_stimulus_event(1).get_port_codes(tempport);
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		if screennum!=0 then
			screentype.set_port_code(screennum);
		end;
		InstructPicWithoutImage.present();NullTrial_wPort.present();
		NullTrial_wPort.get_stimulus_event(1).set_port_code(int(tempport[1]));screentype.set_port_code(0);
		Instruct_text.set_caption(temptext);Instruct_text.redraw();
		int count4 = response_manager.total_response_count(4);
		loop int count3 = response_manager.total_response_count(3) until response_manager.total_response_count(3)>count3 || response_manager.total_response_count(4)>count4 begin end;wait_interval(20); # wait for the right mouse button or space
		if response_manager.total_response_count(4)>count4 then
			wait_interval(100);
			return false;
		else
			wait_interval(100);
			return true;
		end;
		return false;
	end;

	sub bool instructions_with_image (string intsruction_text, int screennum) begin
		string temptext=Instruct_text.caption();
		Instruct_text.set_caption(intsruction_text);Instruct_text.redraw();
		intsruction_text.resize(100);
		NullTrial_wPort.get_stimulus_event(1).set_event_code(intsruction_text);
		array<double> tempport[1];NullTrial_wPort.get_stimulus_event(1).get_port_codes(tempport);
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		if screennum!=0 then
			screentype.set_port_code(screennum);
		end;
		InstructPicWithImage.present();NullTrial_wPort.present();
		NullTrial_wPort.get_stimulus_event(1).set_port_code(int(tempport[1]));screentype.set_port_code(0);
		Instruct_text.set_caption(temptext);Instruct_text.redraw();
		int count4 = response_manager.total_response_count(4);
		loop int count3 = response_manager.total_response_count(3) until response_manager.total_response_count(3)>count3 || response_manager.total_response_count(4)>count4 begin end;wait_interval(20); # wait for the right mouse button or space
		if response_manager.total_response_count(4)>count4 then
			wait_interval(100);
			return false;
		else
			wait_interval(100);
			return true;
		end;
		return false;
	end;

	sub bool check_pseudorandom (array<int,1> arr) begin
		loop int i=4 until i>arr.count()-2 begin
			if (arr[i]-1)/num_images_per_scene==(arr[i+1]-1)/num_images_per_scene || (arr[i]-1)/num_images_per_scene==(arr[i+2]-1)/num_images_per_scene then
				return false;
			end;
			i=i+1;
		end;
		if arr[arr.count()-1]/num_images_per_scene==arr[arr.count()]/num_images_per_scene then
			return false;
		end;
		return true;
	end;

	sub string get_min_sec (int time) begin 
		string str=string(floor(double(mod(time,60000))/1000));
		if str.count()==1 then
			str="0"+str;
		end;
		return string(floor(double(time)/60000))+":"+str;
	end;

		#"We will now start <u>Block #"+string(blocknum)+"</u> out of "+string(scenes_per_block.count()/num_scenes_per_block)+"\nof the spatial task.\n\nGood luck!\n\nRight-click to proceed.",9+blocknum*3-2) then	
	
	sub bool hear_sounds_stories (array<int,1>options, int blocknum) begin
		picture which_images=new picture;
		which_images.set_background_color(255,255,255);
		which_images.add_part(counter,-900,-450);	
		text boxtext=new text;
		string txt1="These locations will feature in <u>Block #"+string(blocknum)+"</u>/"+string(scenes_per_block.count()/num_scenes_per_block)+" of the spatial task.\nIf you want to refresh the story for a location's story, left-click it.\nIf you want to go ahead with the spatial task, right-click anywhere.\n\n\n\n\n\n\n\n\n\n\n\nGood luck!";
		string txt2="These locations will feature in <u>Block #"+string(blocknum)+"</u>/"+string(scenes_per_block.count()/num_scenes_per_block)+" of the spatial task.\nIf you want to refresh the story for a location's story, left-click it.\nIf you want to go ahead with the spatial task, right-click anywhere.\n\n\n\n\n\n\n\n\n\n\n\nLeft click anywhere to stop audio";		
		boxtext.set_caption(txt1);
		boxtext.set_font_size(int(45*scaling_factor));
		boxtext.set_background_color(255,255,255);
		boxtext.set_font_color(0,0,0);
		boxtext.set_formatted_text(true);
		boxtext.redraw();
		which_images.add_part(boxtext,0,-5);
		array<sound>stories[options.count()];
		array<int> ImageLims[options.count()][6];
		string txt="all options are: ";	
		loop int i=1 until i>options.count() begin
			ImageLims[i][1]=-375-scene_width/2;
			ImageLims[i][2]=-375+scene_width/2;
			ImageLims[i][3]=-275-scene_height/2;
			ImageLims[i][4]=-275+scene_height/2;
			ImageLims[i][5]=-375;
			ImageLims[i][6]=-275;
			if i<3 && num_scenes_per_block==4 then
				ImageLims[i][3]=ImageLims[i][3]+350;
				ImageLims[i][4]=ImageLims[i][4]+350;
				ImageLims[i][6]=ImageLims[i][6]+350;
			elseif i<3 && num_scenes_per_block==2 then
				ImageLims[i][3]=ImageLims[i][3]+175;
				ImageLims[i][4]=ImageLims[i][4]+175;
				ImageLims[i][6]=ImageLims[i][6]+175;				
			end;
			if i==2 || i==4 then ImageLims[i][1]=ImageLims[i][1]+750;ImageLims[i][2]=ImageLims[i][2]+750;ImageLims[i][5]=ImageLims[i][5]+750;end;
			#which_images.add_part(scenes[options[i]],ImageLims[i][5],ImageLims[i][6]);
			which_images.add_part(scenes[options[i]],ImageLims[i][5],ImageLims[i][6]);			
			txt=txt+"("+string(options[i])+")"; if i<options.count() then txt=txt+", ";end;
			
			string filename=tempsoundforfilename.filename().replace("stim\\1","log\\S"+SubjectNum+"_"+string(options[i])+"_1");
			int story_length=0;
			string longest_filename=filename;
			loop int idx=2 until !file_exists(filename) begin
				wavefile tmpwav=new wavefile(filename);
				if tmpwav.duration()>story_length then
					longest_filename=filename;
					story_length=tmpwav.duration();
				end;
				filename=filename.replace(string(idx-1)+".wav",string(idx)+".wav");
				idx=idx+1;
			end;				
			stories[i]=new sound(new wavefile(longest_filename));
			stories[i].set_attenuation(0);#0.2
			i=i+1;
		end;

		which_images.add_part(MouseMarker,0,-100);
		which_images.set_part_on_top(which_images.part_count(),true);
		mse.set_xy(0,-100);

		NullTrial_wPort.get_stimulus_event(1).set_event_code(txt);
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		
		which_images.present();
		NullTrial_wPort.present();
		wait_interval(40);			
		NullTrial_wPort.get_stimulus_event(1).set_port_code(9);
		NullTrial_wPort.present();
		wait_interval(25);			
		
		int count4=response_manager.total_response_count(4);
		int time_to_stop=999999999;
		loop int Mouse_on_count = response_manager.total_response_count(1);int Mouse_right_count = response_manager.total_response_count(3);bool first=true until Mouse_right_count<response_manager.total_response_count(3) || count4<response_manager.total_response_count(4) begin
			if first && mse.x()!=0 && mse.y()!=0 then
				NullTrial_wPort.get_stimulus_event(1).set_event_code("Mouse Move");
				array<double> tmp[1];NullTrial_wPort.get_stimulus_event(1).get_port_codes(tmp);
				NullTrial_wPort.get_stimulus_event(1).set_port_code(8); #mouse move code
				NullTrial_wPort.present();
				NullTrial_wPort.get_stimulus_event(1).set_port_code(int(tmp[1]));
				first=false;
			end;
			if response_manager.total_response_count(1)>Mouse_on_count then
				Mouse_on_count=response_manager.total_response_count(1);
				audio_device.stop();
				boxtext.set_caption(txt1);boxtext.redraw();
				time_to_stop=999999999;
				loop int i=1 until i>options.count() begin
					if mse.x()>ImageLims[i][1] && mse.x()<ImageLims[i][2] && mse.y()>ImageLims[i][3] && mse.y()<ImageLims[i][4] then
						NullTrial_wPort.get_stimulus_event(1).set_event_code("Playing story for scene #"+string(options[i]));
						NullTrial_wPort.get_stimulus_event(1).set_port_code(options[i]);
						NullTrial_wPort.present();
						stories[i].present();
						wait_interval(25);			
						boxtext.set_caption(txt2);boxtext.redraw();
						time_to_stop=clock.time()+stories[i].get_wavefile().duration();
						break;
					end;
					i=i+1;
				end;
			end;
			if clock.time()>time_to_stop then
				boxtext.set_caption(txt1);boxtext.redraw();
				time_to_stop=999999999;
			end;
			mse.poll();
			which_images.set_part_x(which_images.part_count(),mse.x());
			which_images.set_part_y(which_images.part_count(),mse.y());
			which_images.present();
		end;
		audio_device.stop();
		default.present();
		wait_interval(500);
		if count4==response_manager.total_response_count(4) then
			return true;
		end;
		return false;
	end;

	sub exposure_only(int stim) begin
		# Subjects are exposed to the objects in their locations; no action is required
		#bitmap img=new bitmap;
		#bitmap scne=new bitmap;
		array<int>parts[2];
		parts[1]=circle_pic[(stim-1)/num_images_per_scene+1].part_count();
		parts[2]=circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count();

		array<int>pos[2];
		sound snd;
		img=objects[item_scene_allocation[stim]];
		img_left.set_filename(img.get_texture(1).filename());
		img_left.load();
		
		int x_shift_for_left_items=-565;
		/*if stim>num_scenes*num_images_per_scene then
			x_shift_for_left_items=-2000;
		end;*/

		pos[1]=true_positions[stim][1];
		pos[2]=true_positions[stim][2];
		item_frame.set_line_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3],255);	item_frame.redraw();
		item_frame_left.set_line_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3],255);	item_frame_left.redraw();
		if (stim-1)/num_images_per_scene+1==3 then
			item_frame.set_line_color(0,0,0,255);	item_frame.redraw();
			item_frame_left.set_line_color(0,0,0,255);	item_frame_left.redraw();
		end;
		circle_pic[(stim-1)/num_images_per_scene+1].add_3dpart(img,double(pos[1]),double(pos[2]),0.0);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(item_frame,pos[1],pos[2]);
		circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),true);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(img_left,x_shift_for_left_items+int((double(pos[1])-(display_device.width()/2-Radius))/4),-250+int((double(pos[2])/Radius)*(display_device.height()/20))+y_shift_left);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(item_frame_left,x_shift_for_left_items+int((double(pos[1])-(display_device.width()/2-Radius))/4),-250+int((double(pos[2])/Radius)*(display_device.height()/20))+y_shift_left);

		ImageStim_for_exposure1.set_stimulus(circle_pic[(stim-1)/num_images_per_scene+1]);
		ImageStim_for_exposure2.set_stimulus(circle_pic[(stim-1)/num_images_per_scene+1]);
		ImageStim_for_exposure3.set_stimulus(circle_pic[(stim-1)/num_images_per_scene+1]);

		snd=sounds[item_scene_allocation[stim]];
		SoundStim_for_exposure1.set_port_code(mod(stim-1,4)+1);
		SoundStim_for_exposure2.set_port_code((stim-1)/num_images_per_scene+1);
		#ImageStim_for_exposure2.set_port_code(stim+3);
		#ImageStim_for_exposure3.set_port_code(22+mod(stim,10));
		SoundStim_for_exposure1.set_stimulus(snd);
		SoundStim_for_exposure2.set_stimulus(snd);
		SoundStim_for_exposure2.set_time(exposure_duration_in_ms-snd.get_wavefile().duration());

		ImageStim_for_exposure2.set_event_code("Exposure Trial - Checkerboard + sound Presentation: Set #"+string(stim/10)+", Picture #="+string(mod(stim,10))+"/4, True Location(x,y)=("+string(pos[1])+","+string(pos[2])+")");
		exposure_trial.present();
		
		loop until circle_pic[(stim-1)/num_images_per_scene+1].part_count()==parts[1] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_part(circle_pic[(stim-1)/num_images_per_scene+1].part_count());end;
		loop until circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count()==parts[2] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_3dpart(circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count());end;
		
		#loop int count = response_manager.total_response_count(3) until response_manager.total_response_count(3) > count begin end; wait_interval(5); # wait for the right mouse button press # currently inactive because exposure is self paced
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Exposure Trial; End Trial");
		array<double> tmp[1];NullTrial_wPort.get_stimulus_event(1).get_port_codes(tmp);
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		NullTrial_wPort.present();default.present();
		NullTrial_wPort.get_stimulus_event(1).set_port_code(int(tmp[1]));
		wait_interval(exposure_ITI);
	end;

	sub array<int,1> positioning_only (int stim, bool feedback, bool random_starting_point,bool context_quest, bool was_previous_correct) begin #(array<int,1> stim,bool feedback,bool with_sound, bool random_starting_point) begin
		# Subject is required to position the stimuli in their correct locations
		#bitmap img=new bitmap;
		#bitmap scne=new bitmap;
		mse.set_min_max( 1, -pic_side_size, display_device.width() / 2);
		array<int>parts[4];
		parts[1]=circle_pic[(stim-1)/num_images_per_scene+1].part_count();
		parts[2]=circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count();
		parts[3]=no_circle_pic[(stim-1)/num_images_per_scene+1].part_count();
		parts[4]=no_circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count();
		array<int>pos[2];
		sound snd;
		img=objects[item_scene_allocation[stim]];
		img_left.set_filename(img.get_texture(1).filename());
		img_left.load();
		pos[1]=true_positions[stim][1];
		pos[2]=true_positions[stim][2];
		arrow_graphic arrow=new arrow_graphic();
		arrow.set_color(rgb_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3]));
		arrow_graphic arrow_inner=new arrow_graphic();
		arrow_inner.set_color(255,255,255,255);
		arrow_graphic arrow_left=new arrow_graphic();
		arrow_left.set_color(rgb_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3]));
		arrow_graphic arrow_inner_left=new arrow_graphic();
		arrow_inner_left.set_color(255,255,255,255);

		item_frame.set_line_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3],255);	item_frame.redraw();
		item_frame_left.set_line_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3],255);	item_frame_left.redraw();
		if (stim-1)/num_images_per_scene+1==3 then
			item_frame.set_line_color(0,0,0,255);	item_frame.redraw();
			item_frame_left.set_line_color(0,0,0,255);	item_frame_left.redraw();
			arrow.set_color(rgb_color(0,0,0));
			arrow_left.set_color(rgb_color(0,0,0));
		end;		
		array<int>user_positions[2]; #current position of the image, starts at 0,0
		if random_starting_point then
			int min_distance_from_true_position=correct_incorrect_criterion;#100;#pixels
			loop bool crit=false until crit==true begin
				user_positions[1]=random(-Radius,Radius)+display_device.width()/2-Radius;
				user_positions[2]=random(-Radius,Radius);#0;
				if pow(pow(user_positions[1]-(display_device.width()/2-Radius),2)+pow(user_positions[2],2),0.5)<Radius-margin && pow(pow(user_positions[1]-(display_device.width()/2-Radius),2)+pow(user_positions[2],2),0.5)>inner_clear_Radius && pow(pow(user_positions[1]-pos[1],2)+pow(user_positions[2]-pos[2],2),0.5)>min_distance_from_true_position then
					crit=true; #this checks that the position is not outside the circle or inside the innter circle (and not too close to the real location)
				end;
			end;
		end;
		int x_shift_for_left_items=-565;
		/*if stim>num_scenes*num_images_per_scene then
			x_shift_for_left_items=-2000;
		end;*/
				
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(img_left,x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4),-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(item_frame_left,x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4),-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left);
		circle_pic[(stim-1)/num_images_per_scene+1].add_3dpart(img,double(user_positions[1]),double(user_positions[2]),0.0);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(item_frame,user_positions[1],user_positions[2]);
		circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),true);
		array<int> tmp_user_positions[]=user_positions;

		Instruct_text.set_caption("Drag item, then right click");
		#Instruct_text.set_caption("Move the item to its\ncorrect location and hit\nthe right mouse button");
		Instruct_text.redraw();
		#circle_pic[(stim-1)/num_images_per_scene+1].add_part(Instruct_text,-display_device.width()/4,2.25*display_device.height()/5+y_shift_left);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(Instruct_text,-Radius-25,display_device.height()/2-40+y_shift_left/2);
		#circle_pic[(stim-1)/num_images_per_scene+1].add_part(MouseMarker,0,0);
		circle_pic[(stim-1)/num_images_per_scene+1].add_part(MouseMarker,user_positions[1],user_positions[2]);
		circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),true);
		array<double> tempport[1];NullTrial_wPort.get_stimulus_event(1).get_port_codes(tempport);
		if context_quest then
			int questnum=random(1,Questions.count());
			picture picsceneonly=new picture;
			picsceneonly.set_background_color(colors[(stim-1)/num_images_per_scene+1][1],colors[(stim-1)/num_images_per_scene+1][2],colors[(stim-1)/num_images_per_scene+1][3]);
			if (stim-1)/num_images_per_scene+1==3 then
				picsceneonly.set_background_color(0,0,0);
			end;
			scenes[(stim-1)/num_images_per_scene+1].set_load_size(scene_height*2,scene_width*2,0);scenes[(stim-1)/num_images_per_scene+1].load();
			picsceneonly.add_part(scenes[(stim-1)/num_images_per_scene+1],0,0);
			NullTrial_wPort.get_stimulus_event(1).set_port_code((stim-1)/num_images_per_scene+1);
			NullTrial_wPort.get_stimulus_event(1).set_event_code("Positioning Trial - Checkerboard: Scene #"+string((stim-1)/num_images_per_scene+1)+", Object #="+string(mod(stim-1,4)+1)+", True Location(x,y)=("+string(pos[1])+","+string(pos[2])+"), Context quest onset, Question: "+Questions[questnum]);
			NullTrial_wPort.present();picsceneonly.present();
			scenes[(stim-1)/num_images_per_scene+1].set_load_size(scene_height,scene_width,0);scenes[(stim-1)/num_images_per_scene+1].load();
			text quest=new text();
			quest.set_caption(Questions[questnum]+"\n\n\n\n\n\n\n\n\n\n<<No<<           >>Yes>>");
			quest.set_font_size(int(60*scaling_factor));
			quest.set_background_color(255,255,255,0);
			quest.set_font_color(0,0,0);
			quest.redraw();
			img=objects[item_scene_allocation[stim]];
			img.set_size(double(pic_side_size*4),double(pic_side_size*4),0.0);
			no_circle_pic[(stim-1)/num_images_per_scene+1].add_3dpart(img,double(display_device.width()/2-Radius),0.0,0.0);
			no_circle_pic[(stim-1)/num_images_per_scene+1].add_part(quest,display_device.width()/2-Radius,0);
			wait_interval(YNDelay);
			no_circle_pic[(stim-1)/num_images_per_scene+1].present();
			NullTrial_wPort.present();
			wait_interval(25);
			int count1 = response_manager.total_response_count(1);int count3 = response_manager.total_response_count(3);
			loop until response_manager.total_response_count(3)>count3 || response_manager.total_response_count(1)>count1 begin end;wait_interval(20); # wait for the right mouse button or space
			if Answers[(stim-1)/num_images_per_scene+1][questnum][mod(stim-1,num_images_per_scene)+1]!=0 then
				if (Answers[(stim-1)/num_images_per_scene+1][questnum][mod(stim-1,num_images_per_scene)+1]==-1 && response_manager.total_response_count(1)>count1) || (Answers[(stim-1)/num_images_per_scene+1][questnum][mod(stim-1,num_images_per_scene)+1]==1 && response_manager.total_response_count(3)>count3) then
					# Correct answer	
					NullTrial_wPort.get_stimulus_event(1).set_event_code("Correct answer");
					NullTrial_wPort.present();
				else
					# Incorrect answer	
					NullTrial_wPort.get_stimulus_event(1).set_event_code("Incorrect answer");
					Instruct_text.set_caption("Wrong answer!\nPlease try and vividly recall your story");Instruct_text.redraw();
					InstructPicWithoutImage.present();NullTrial_wPort.present();
					wait_interval(YNDelay);
					default.present();
					wait_interval(positioning_ITI);
					array<int>tmpreturn[]={-1,-1,-1};
					loop until circle_pic[(stim-1)/num_images_per_scene+1].part_count()==parts[1] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_part(circle_pic[(stim-1)/num_images_per_scene+1].part_count());end; #clean up image		
					loop until circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count()==parts[2] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_3dpart(circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count());end;
					loop until no_circle_pic[(stim-1)/num_images_per_scene+1].part_count()==parts[3] begin no_circle_pic[(stim-1)/num_images_per_scene+1].remove_part(no_circle_pic[(stim-1)/num_images_per_scene+1].part_count());end; #clean up image		
					loop until no_circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count()==parts[4] begin no_circle_pic[(stim-1)/num_images_per_scene+1].remove_3dpart(no_circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count());end; #clean up image		
					out_results.print("-1\t-1\t-1\t1\t"+img.get_texture(1).filename()+"\n");
					term.print_line("-1\t\t-1\t\t-1\t1\t\t"+img.get_texture(1).filename());
					return tmpreturn;
				end;
			end;
			
			no_circle_pic[(stim-1)/num_images_per_scene+1].remove_part(no_circle_pic[(stim-1)/num_images_per_scene+1].part_count());
			int num_steps=50;
			array<double>motion_steps[3][num_steps];
			motion_steps[1][1]=display_device.width()/2-Radius;motion_steps[2][1]=0;motion_steps[3][1]=pic_side_size*4;
			array<double>steps[3];
			steps[1]=double(display_device.width()/2-Radius-user_positions[1])/(num_steps-1);
			steps[2]=double(-user_positions[2])/(num_steps-1);
			steps[3]=double(pic_side_size*4-pic_side_size)/(num_steps-1);
			loop int i=2 until i>num_steps begin
				loop int j=1 until j>3 begin
					motion_steps[j][i]=motion_steps[j][i-1]-steps[j]; j=j+1;
				end;
				i=i+1;
			end;
			loop int i=1 until i>num_steps begin
				img.set_size(motion_steps[3][i],motion_steps[3][i],0);
				no_circle_pic[(stim-1)/num_images_per_scene+1].set_3dpart_xyz(no_circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count(),motion_steps[1][i],motion_steps[2][i],0.0);
				no_circle_pic[(stim-1)/num_images_per_scene+1].present();
				i=i+1;
			end;
		else
			NullTrial_wPort.get_stimulus_event(1).set_port_code((stim-1)/num_images_per_scene+1);
			NullTrial_wPort.get_stimulus_event(1).set_event_code("Positioning Trial - Checkerboard: Scene #"+string((stim-1)/num_images_per_scene+1)+", Object #="+string(mod(stim-1,4)+1)+", True Location(x,y)=("+string(pos[1])+","+string(pos[2])+"), Filler (for port code)");			
			NullTrial_wPort.present();
			wait_interval(25);
		end;
		int num_port_code=0;
		string txt_event_code;
		snd=sounds[item_scene_allocation[stim]];
		NullTrial_wPort.get_stimulus_event(1).set_port_code(mod(stim-1,4)+1);
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Positioning Trial - Checkerboard: Scene #"+string((stim-1)/num_images_per_scene+1)+", Object #="+string(mod(stim-1,4)+1)+", True Location(x,y)=("+string(pos[1])+","+string(pos[2])+"), Positioning onset and sound");
		int timetmp=clock.time();
		snd.present();
		NullTrial_wPort.present();
		circle_pic[(stim-1)/num_images_per_scene+1].present();
		wait_interval(25);
		#mse.set_xy(0,0);
		mse.set_xy(user_positions[1],user_positions[2]);
		loop
			int Mouse_on_count = response_manager.total_response_count(1);
			int Right_mouse_count = response_manager.total_response_count(3);
			bool first=true; # to mark first mouse motion
			bool item_not_left_in_the_center;
			bool scene_presented;
		until
			response_manager.total_response_count(3) > Right_mouse_count && item_not_left_in_the_center # until the right mouse button is pressed after the target has been moved
		begin
			if first && mse.x()!=user_positions[1] && mse.y()!=user_positions[2] then
				NullTrial_wPort.get_stimulus_event(1).set_event_code("Mouse Move");
				NullTrial_wPort.get_stimulus_event(1).set_port_code(8); #mouse move code
				NullTrial_wPort.present();
				NullTrial_wPort.get_stimulus_event(1).set_event_code("Positioning Trial - Checkerboard: Scene #"+string((stim-1)/num_images_per_scene+1)+", Object #="+string(mod(stim-1,4)+1)+", True Location(x,y)=("+string(pos[1])+","+string(pos[2])+"), Positioning onset and sound");
				first=false;
			end;
			if response_manager.total_response_count(3) > Right_mouse_count then
				Right_mouse_count=response_manager.total_response_count(3);
				Instruct_text.set_caption("<b>Drag item</b>, then right click");
				Instruct_text.redraw();
			end;
			if response_manager.total_response_count(1)>Mouse_on_count then
				int Mouse_off_count = response_manager.total_response_count(2);
				array<int> point_chosen[2];
				Mouse_on_count=response_manager.total_response_count(1);
				bool target_chosen=false;
				
				if mse.x()>user_positions[1]-pic_side_size/2 && mse.x()<user_positions[1]+pic_side_size/2 && mse.y()>user_positions[2]-pic_side_size/2 && mse.y()<user_positions[2]+pic_side_size/2 then
					item_not_left_in_the_center=true;
					target_chosen=true;
					point_chosen[1]=mse.x()-user_positions[1];
					point_chosen[2]=mse.y()-user_positions[2];
					#snd.present();
					#NullTrial_wPort.present();
				end;
				
				loop until response_manager.total_response_count(2)>Mouse_off_count begin
					mse.poll();
					circle_pic[(stim-1)/num_images_per_scene+1].set_part_x(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),mse.x());
					circle_pic[(stim-1)/num_images_per_scene+1].set_part_y(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),mse.y());
					if first && mse.x()!=user_positions[1] && mse.y()!=user_positions[2] then
						NullTrial_wPort.get_stimulus_event(1).set_event_code("Mouse Move");
						NullTrial_wPort.get_stimulus_event(1).set_port_code(8);#mouse move code
						NullTrial_wPort.present();
						if num_port_code!=0 then
							NullTrial_wPort.get_stimulus_event(1).set_port_code(mod(stim-1,4)+1);
						end;
						NullTrial_wPort.get_stimulus_event(1).set_event_code("Positioning Trial - Checkerboard: Scene #"+string((stim-1)/num_images_per_scene+1)+", Object #="+string(mod(stim-1,4)+1)+", True Location(x,y)=("+string(pos[1])+","+string(pos[2])+"), Positioning onset and sound");
						first=false;
					end;
					if target_chosen then
						user_positions[1]=mse.x()-point_chosen[1];
						user_positions[2]=mse.y()-point_chosen[2];
						NullTrial_woPort.get_stimulus_event(1).set_event_code("Current target location(x,y)=("+string(user_positions[1])+","+string(user_positions[2])+")");
						NullTrial_woPort.present();
						circle_pic[(stim-1)/num_images_per_scene+1].set_3dpart_xyz(circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count(),double(user_positions[1]),double(user_positions[2]),0.0);
						circle_pic[(stim-1)/num_images_per_scene+1].set_part_x(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-2,user_positions[1]);
						circle_pic[(stim-1)/num_images_per_scene+1].set_part_y(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-2,user_positions[2]);
						circle_pic[(stim-1)/num_images_per_scene+1].set_part_x(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-3,x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4));
						circle_pic[(stim-1)/num_images_per_scene+1].set_part_y(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-3,-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left);
						circle_pic[(stim-1)/num_images_per_scene+1].set_part_x(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-4,x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4));
						circle_pic[(stim-1)/num_images_per_scene+1].set_part_y(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-4,-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left);
					end;
					circle_pic[(stim-1)/num_images_per_scene+1].present();
				end;
				#term.print(round(pow(pow(pos[1]-user_positions[1],2)+pow(pos[2]-user_positions[2],2),0.5),2));
				#term.print(user_positions);
			end;
			mse.poll();
			circle_pic[(stim-1)/num_images_per_scene+1].set_part_x(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),mse.x());
			circle_pic[(stim-1)/num_images_per_scene+1].set_part_y(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),mse.y());
			circle_pic[(stim-1)/num_images_per_scene+1].present();
		end;
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Right mouse button pressed; Final Chosen Location(x,y)=("+string(user_positions[1])+","+string(user_positions[2])+"); End Trial");
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		double dist=round(pow(pow(pos[1]-user_positions[1],2)+pow(pos[2]-user_positions[2],2),0.5),2);
		out_results.print(string(user_positions[1])+"\t"+string(user_positions[2])+"\t"+string(int(dist))+"\t1\t"+img.get_texture(1).filename()+"\n");
		term.print_line(string(user_positions[1])+"\t\t"+string(user_positions[2])+"\t\t"+string(int(dist))+"\t1\t\t"+img.get_texture(1).filename());
		if feedback==true then
			loop until circle_pic[(stim-1)/num_images_per_scene+1].part_count()==parts[1] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_part(circle_pic[(stim-1)/num_images_per_scene+1].part_count());end; #clean up image		
			loop until circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count()==parts[2] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_3dpart(circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count());end;
			cuboid img2=new cuboid(double(pic_side_size),double(pic_side_size),0.0);
			texture textur=new texture();
			textur.set_filename(img.get_texture(1).filename());	textur.load();
			textur.set_alpha(175);
			img2.set_texture(textur);
			img2.set_emissive(1.0,1.0,1.0);		
			circle_pic[(stim-1)/num_images_per_scene+1].add_3dpart(img2,double(user_positions[1]),double(user_positions[2]),0.0);		
			circle_pic[(stim-1)/num_images_per_scene+1].add_3dpart(img,double(pos[1]),double(pos[2]),0.0);
			circle_pic[(stim-1)/num_images_per_scene+1].add_part(item_frame,pos[1],pos[2]);
			circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),true);

			circle_pic[(stim-1)/num_images_per_scene+1].add_part(img_left,x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4),-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left);
			circle_pic[(stim-1)/num_images_per_scene+1].add_part(img_left,x_shift_for_left_items+int((double(pos[1])-(display_device.width()/2-Radius))/4),-250+int((double(pos[2])/Radius)*(display_device.height()/20))+y_shift_left);
			circle_pic[(stim-1)/num_images_per_scene+1].add_part(item_frame_left,x_shift_for_left_items+int((double(pos[1])-(display_device.width()/2-Radius))/4),-250+int((double(pos[2])/Radius)*(display_device.height()/20))+y_shift_left);
			
			Instruct_text.set_caption(" ");#Instruct_text.set_caption("Hit the right mouse button to continue");
			Instruct_text.redraw();
			int animation_time=0;
			if int(dist)>correct_incorrect_criterion then
				arrow.set_coordinates(user_positions[1],user_positions[2],pos[1],pos[2]);
				arrow_inner.set_coordinates(user_positions[1],user_positions[2],pos[1],pos[2]);
				arrow_inner_left.set_coordinates(x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4),-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left,-565+int((double(pos[1])-(display_device.width()/2-Radius))/4),-250+int((double(pos[2])/Radius)*(display_device.height()/20))+y_shift_left);
				arrow_left.set_coordinates(x_shift_for_left_items+int((double(user_positions[1])-(display_device.width()/2-Radius))/4),-250+int((double(user_positions[2])/Radius)*(display_device.height()/20))+y_shift_left,x_shift_for_left_items+int((double(pos[1])-(display_device.width()/2-Radius))/4),-250+int((double(pos[2])/Radius)*(display_device.height()/20))+y_shift_left);
				arrow.set_line_width(15); arrow_inner.set_line_width(10); arrow_left.set_line_width(6); arrow_inner_left.set_line_width(4); 
				arrow.set_head_width(45); arrow_inner.set_head_width(30); arrow_left.set_head_width(18); arrow_inner_left.set_head_width(12); 
				arrow.set_head_length(45); arrow_inner.set_head_length(42); arrow_left.set_head_length(18); arrow_inner_left.set_head_length(17); 
				arrow.redraw(); arrow_inner.redraw(); arrow_left.redraw(); arrow_inner_left.redraw();
				circle_pic[(stim-1)/num_images_per_scene+1].add_part(arrow,0,0);
				circle_pic[(stim-1)/num_images_per_scene+1].add_part(arrow_inner,0,0);
				circle_pic[(stim-1)/num_images_per_scene+1].add_part(arrow_left,0,0);
				circle_pic[(stim-1)/num_images_per_scene+1].add_part(arrow_inner_left,0,0);
				circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-3,true);
				circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-2,true);
				circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count()-1,true);
				circle_pic[(stim-1)/num_images_per_scene+1].set_part_on_top(circle_pic[(stim-1)/num_images_per_scene+1].part_count(),true);
			elseif was_previous_correct then
				animation_time=200;
				# designed for single digit num correct (<=8)
				int num_correct=int(status_txt.caption().substring(1,1));
				text new_score=new text;
				new_score.set_caption(string(num_correct+1)+status_txt.caption().substring(2,2)+"\n \n ");
				new_score.set_font_size(int(50*scaling_factor));
				new_score.set_background_color(255,255,255,0);
				new_score.set_font_color(0,150,0);
				new_score.redraw();
				status_txt.set_caption(status_txt.caption().substring(4,status_txt.caption().count()-3));status_txt.redraw();
				circle_pic[(stim-1)/num_images_per_scene+1].add_part(new_score,display_device.width()/2-100,-display_device.height()/2+100);
				loop int j=30+animation_time/10-1 until j==30 begin
					new_score.set_font_size(int(j*scaling_factor));new_score.redraw();
					j=j-1;
					circle_pic[(stim-1)/num_images_per_scene+1].present();
					wait_interval(10);
				end;			
			end;
			circle_pic[(stim-1)/num_images_per_scene+1].present();
			NullTrial_wPort.present();
			#loop int count = response_manager.total_response_count(3) until response_manager.total_response_count(3) > count begin end; wait_interval(5); # wait for the right mouse button press
			wait_interval(feedback_duration_in_ms-snd.get_wavefile().duration()-animation_time);
			snd.present();
			NullTrial_wPort.present();
			wait_interval(snd.get_wavefile().duration());
		else			
			wait_interval(50);
		end;
		
		loop until circle_pic[(stim-1)/num_images_per_scene+1].part_count()==parts[1] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_part(circle_pic[(stim-1)/num_images_per_scene+1].part_count());end; #clean up image		
		loop until circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count()==parts[2] begin circle_pic[(stim-1)/num_images_per_scene+1].remove_3dpart(circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count());end;
		loop until no_circle_pic[(stim-1)/num_images_per_scene+1].part_count()==parts[3] begin no_circle_pic[(stim-1)/num_images_per_scene+1].remove_part(no_circle_pic[(stim-1)/num_images_per_scene+1].part_count());end; #clean up image		
		loop until no_circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count()==parts[4] begin no_circle_pic[(stim-1)/num_images_per_scene+1].remove_3dpart(no_circle_pic[(stim-1)/num_images_per_scene+1].d3d_part_count());end; #clean up image		
		
		default.present();
		wait_interval(positioning_ITI);
		user_positions.add(int(dist));
		mse.set_min_max( 1, -display_device.width() / 2, display_device.width() / 2);
		return user_positions;
	end;

/*	sub array<int,1> shuffle_with_no_consec(array<int,1> arr1) begin
		array<int>ord[arr1.count()];ord.fill(1,ord.count(),1,1);
		bool cond=true;
		loop until !cond begin
			cond=false;
			ord.shuffle();
			loop int idx=1; int i=1 until i==arr1.count() || idx>1000 begin
				loop until arr1[ord[i]]!=arr1[ord[i+1]] begin
					ord.shuffle(i+1,arr1.count());
					idx=idx+1;
					if idx>1000 then
						cond=true;
						break;
					end;
				end;
				i=i+1;
			end;
		end;
		return ord;
	end;			*/

	sub array<int,1> shuffle_with_no_consec(array<int,1> arr1) begin
		array<int>ord[arr1.count()];ord.fill(1,ord.count(),1,1);
		bool cond=true;
		loop int numtrys=1 until !cond || numtrys>100 begin
			cond=false;
			ord.shuffle();
			loop int idx=1; int i=1 until i==arr1.count() || idx>100 begin
				loop until (arr1[ord[i]]-1)/num_images_per_scene!=(arr1[ord[i+1]]-1)/num_images_per_scene begin
					ord.shuffle(i+1,arr1.count());
					idx=idx+1;
					if idx>100 then
						cond=true;
						numtrys=numtrys+1;
						break;
					end;
				end;
				i=i+1;
			end;
		end;
		array<int>return_array[ord.count()];
		loop int i=1 until i>ord.count() begin
			return_array[i]=arr1[ord[i]];
			i=i+1;
		end;
		return return_array;
	end;			
		
	sub array<int,1> sort(array<int,1> arr) begin
		array<int>sorted[]={1};
		array<int>vals[1];vals[1]=arr[1];
		loop int i=2 until i>num_scenes begin
			int j=1;
			loop until j>vals.count() begin
				if arr[i]<vals[j] then break; end;
				j=j+1;
			end;
			if j>vals.count() then
				vals.add(arr[i]);
				sorted.add(i);
			else
				vals.resize(vals.count()+1);
				sorted.resize(sorted.count()+1);
				loop int n=vals.count() until n==j begin
					vals[n]=vals[n-1];
					sorted[n]=sorted[n-1];
					n=n-1;
				end;
				vals[j]=arr[i];
				sorted[j]=i;
			end;
			i=i+1;
		end;
		term.print_line(vals);
		# taking care of equal values (randomizing them)
		loop int i=1 until i>vals.count() begin
			int j=i;
			loop until j>vals.count() begin
				if vals[j]!=vals[i] then break; end;
				j=j+1;
			end;
			j=j-1;
			if j-i>0 then
				array<int>rnd[j-i+1];rnd.fill(1,rnd.count(),1,1);rnd.shuffle();
				loop int n=1 until n>rnd.count() begin
					rnd[n]=sorted[i+rnd[n]-1];
					n=n+1;
				end;
				loop int n=1 until n>rnd.count() begin
					sorted[i+n-1]=rnd[n];
					n=n+1;
				end;
			end;
			i=j+1;
		end;
		return sorted;
	end;


	sub string scene_item_recall(int scene_num, int time_limit_for_func)
	begin
		# Subject is required to recall which items were related to the scene
		picture which_images=new picture;
		which_images.set_background_color(255,255,255);
		loop int i=1 until i>4 begin loop int j=1 until j>2 begin box_square[i][j]=unit_square[i][j]*box_size/(j*2); j=j+1; end; i=i+1; end;
		#which_images.add_part(scenes[scene_num],0,display_device.height()/2-200);
		which_images.add_part(scenes[scene_num],0,display_device.height()/2-200);
		text instructText=new text;
		instructText.set_caption("Try to recall all "+string(num_images_per_scene)+"\nassociated objects.\nTime left: "+string(time_limit_for_func)+"\nHit Esc to continue.");
		instructText.set_formatted_text(true);
		instructText.set_font_size(int(35*scaling_factor));
		instructText.set_background_color(255,255,255);
		instructText.set_font_color(0,0,0);
		instructText.redraw();
		which_images.add_part(instructText,display_device.width()/4+100,2*display_device.height()/5-50);
		loop int i=1 until i>4 begin box_square[i][1]=unit_square[i][1]*box_size*2; box_square[i][2]=unit_square[i][2]*box_size*1; i=i+1; end;
		text input_txt=new text;
		input_txt.set_caption(" ");
		input_txt.ALIGN_LEFT;
		input_txt.set_font_size(int(45*scaling_factor));
		input_txt.set_background_color(255,255,255);
		input_txt.set_font_color(0,0,0);
		input_txt.redraw();
		line_graphic box_for_input=new line_graphic();
		box_for_input.set_line_color(0,0,0,255);
		box_for_input.set_fill_color(255,255,255,0);
		box_for_input.set_line_width(2);
		box_for_input.add_polygon(box_square,true,1,0);
		box_for_input.redraw();
		which_images.add_part(box_for_input,0,-200);
		which_images.add_part(input_txt,0,-200);	
		system_keyboard.set_log_keypresses( true );
		system_keyboard.set_delimiter( 8 );
		system_keyboard.set_max_length( 1 );
		system_keyboard.set_time_out(500);
		int start_time=clock.time();
		loop string caption = "" until clock.time()>time_limit_for_func*1000+start_time
		begin
			if caption.count() <= 0 then
				input_txt.set_caption( " ", true );
			else
				input_txt.set_caption( caption, true );
			end;
			which_images.present();
			string press = system_keyboard.get_input();
			instructText.set_caption("Try to recall all "+string(num_images_per_scene)+"\nassociated objects.\nTime left: "+string(((time_limit_for_func*1000+start_time)-clock.time())/1000)+"\nHit Esc to continue.");instructText.redraw();			
			if system_keyboard.last_input_type() == system_keyboard.DELIMITER then
				if caption.count() > 0 then
					caption.resize( caption.count() - 1 );
				end;
			elseif press=="" then
				start_time=-(time_limit_for_func*1000);
			else 
				caption = caption + press;
			end;
		end;
		return input_txt.caption();
	end;
		

	sub int was_sound_played_in_sleep_or_not begin
		# Subject is required to choose if the presented sounds were played or not during sleep
		picture rem_not_sound=new picture;
		rem_not_sound.set_background_color(255,255,255);
		int box_size2=300; # size of boxes for new/old test and color test
		box_size2=int(box_size2*scaling_factor);
		array<double>unit_square2[][]={{-1.0,-1},{-1,1},{1,1},{1,-1}}; #Used to set the frame square around the image
		array<double>box_square2[4][2];
		loop int i=1 until i>4 begin loop int j=1 until j>2 begin box_square2[i][j]=unit_square2[i][j]*box_size2/2; j=j+1; end; i=i+1; end;
		loop int i=1 until i>2 begin
			text boxtext=new text;
			if i==1 then
				boxtext.set_caption("Don't\nremember\nit played");
			else
				boxtext.set_caption("Remember\nit played");
			end;
			boxtext.set_font_size(int(50*scaling_factor));
			boxtext.set_background_color(255,255,255);
			boxtext.set_font_color(0,0,0);
			boxtext.redraw();
			line_graphic box_for_text=new line_graphic();
			box_for_text.set_line_color(0,0,0,255);
			box_for_text.set_fill_color(255,255,255,0);
			box_for_text.set_line_width(2);
			box_for_text.add_polygon(box_square2,true,1,0);
			box_for_text.redraw();
			rem_not_sound.add_part(box_for_text,(i+1)*display_device.width()/5-display_device.width()/2,-display_device.height()/4);
			rem_not_sound.add_part(boxtext,(i+1)*display_device.width()/5-display_device.width()/2,-display_device.height()/4);
			i=i+1;
		end;
		
		Instruct_text.set_caption("Do you remember hearing this sound during sleep?");
		Instruct_text.set_font_size(int(50*scaling_factor));
		Instruct_text.redraw();
		rem_not_sound.add_part(Instruct_text,0,display_device.height()/3);		

		rem_not_sound.add_part(MouseMarker,0,0);
		rem_not_sound.set_part_on_top(rem_not_sound.part_count(),true);
		mse.set_xy(0,0);

		NullTrial_woPort.get_stimulus_event(1).set_event_code("Remember sounds during sleep");
		rem_not_sound.present();
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Rem sounds");
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		NullTrial_wPort.present();
		
		int chosen_option;
		loop
			int Mouse_on_count = response_manager.total_response_count(1);
			bool first=true; # to mark first mouse motion
		until
			chosen_option!=0
		begin
			if first && mse.x()!=0 && mse.y()!=0 then
				NullTrial_wPort.get_stimulus_event(1).set_event_code("Mouse Move");
				NullTrial_wPort.get_stimulus_event(1).set_port_code(8); #mouse move code
				NullTrial_wPort.present();
				first=false;
			end;
			if response_manager.total_response_count(1)>Mouse_on_count then
				int Mouse_off_count = response_manager.total_response_count(2);
				array<int> point_chosen[2];
				Mouse_on_count=response_manager.total_response_count(1);
				if mse.y()>-display_device.height()/4-box_size2/2 && mse.y()<-display_device.height()/4+box_size2/2 then
					loop int i=1 until i>2 begin
						if mse.x()>(i+1)*display_device.width()/5-display_device.width()/2-box_size2/2 && mse.x()<(i+1)*display_device.width()/5-display_device.width()/2+box_size2/2 then
							chosen_option=i;
							break;
						end;
						i=i+1;
					end;
				end;
				
				loop until response_manager.total_response_count(2)>Mouse_off_count begin
					mse.poll();
					rem_not_sound.set_part_x(rem_not_sound.part_count(),mse.x());
					rem_not_sound.set_part_y(rem_not_sound.part_count(),mse.y());					
					rem_not_sound.present();
				end;
			end;
			mse.poll();
			rem_not_sound.set_part_x(rem_not_sound.part_count(),mse.x());
			rem_not_sound.set_part_y(rem_not_sound.part_count(),mse.y());
			rem_not_sound.present();
		end;
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Chosen don't\do remember = "+string(chosen_option)+"; End Trial");
		NullTrial_wPort.get_stimulus_event(1).set_port_code(28);
		NullTrial_wPort.present();
		default.present();wait_interval(500);
		Instruct_text.set_font_size(int(40*scaling_factor));
		return chosen_option;
	end;

	sub array<int,1> record_answers(int scene, int questionnum) begin
		array<int>item_order[num_images_per_scene];
		item_order.fill(1,item_order.count(),1,1);
		item_order.shuffle();
		array<double>image_pos_quiz[4][2]={{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}};
		image_pos_quiz[1][1]=double(-2*display_device.width()/6);image_pos_quiz[2][1]=double(-2*display_device.width()/6);
		image_pos_quiz[3][1]=double(display_device.width()/6);image_pos_quiz[4][1]=double(display_device.width()/6);
		image_pos_quiz[1][2]=double(display_device.height()/6);image_pos_quiz[3][2]=double(display_device.height()/6);
		image_pos_quiz[2][2]=-double(display_device.height()/5);image_pos_quiz[4][2]=-double(display_device.height()/5);

		picture quiz=new picture();
		quiz.set_background_color(255,255,255);
		array<double>square[][]={{-1.0,-1},{-1,1},{1,1},{1,-1}};
		loop int i=1 until i>4 begin square[i][1]=square[i][1]*pic_side_size/2; square[i][2]=square[i][2]*pic_side_size/2; i=i+1; end;	
		text yesbox=new text;
		yesbox.set_font_size(int(40*scaling_factor));
		yesbox.set_background_color(255,255,255,0);
		yesbox.set_font_color(0,0,0);
		yesbox.set_caption("Yes");
		yesbox.redraw();
		text nobox=new text;
		nobox.set_font_size(int(40*scaling_factor));
		nobox.set_background_color(255,255,255,0);
		nobox.set_font_color(0,0,0);
		nobox.set_caption("No");
		nobox.redraw();
		line_graphic box_for_text=new line_graphic();
		box_for_text.set_line_color(0,0,0,255);
		box_for_text.set_fill_color(255,255,255,0);
		box_for_text.set_line_width(2);
		box_for_text.add_polygon(square,true,1,0);
		box_for_text.redraw();
		line_graphic whitebox=new line_graphic();
		whitebox.set_line_color(0,0,0,255);
		whitebox.set_fill_color(255,255,255,255);
		whitebox.set_line_width(2);
		whitebox.add_polygon(square,true,1,0);
		whitebox.redraw();
		line_graphic yellowbox=new line_graphic();
		yellowbox.set_line_color(0,0,0,255);
		yellowbox.set_fill_color(255,255,0,255);
		yellowbox.set_line_width(2);
		yellowbox.add_polygon(square,true,1,0);
		yellowbox.redraw();
		
		loop int i=1 until i>item_order.count() begin
			quiz.add_3dpart(objects[item_scene_allocation[(scene-1)*num_images_per_scene+item_order[i]]],image_pos_quiz[i][1],image_pos_quiz[i][2],0.0);	
			quiz.add_part(nobox,image_pos_quiz[i][1]+pic_side_size*2.5,image_pos_quiz[i][2]);
			quiz.add_part(yesbox,image_pos_quiz[i][1]+pic_side_size*4,image_pos_quiz[i][2]);
			quiz.add_part(box_for_text,image_pos_quiz[i][1]+pic_side_size*2.5,image_pos_quiz[i][2]);
			quiz.add_part(box_for_text,image_pos_quiz[i][1]+pic_side_size*4,image_pos_quiz[i][2]);
			i=i+1;
		end;	
		text continue_text=new text;continue_text.set_background_color(255,255,255);continue_text.set_font_color(0,0,0);continue_text.set_font_size(int(40*scaling_factor));
		continue_text.set_caption("Right click to continue");
		continue_text.redraw();
		Instruct_text.set_caption(Questions[questionnum]);
		Instruct_text.redraw();
		quiz.add_part(Instruct_text,0,display_device.height()*5/12);		
		quiz.add_part(MouseMarker,0,0);
		quiz.set_part_on_top(quiz.part_count(),true);
		mse.set_xy(0,0);
	
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Quiz: Scene #"+string(scene)+", Question #"+string(questionnum));
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		screentype.set_port_code(2);
		quiz.present();
		NullTrial_woPort.present();
		screentype.set_port_code(0);
		array<int>answers[item_order.count()];
		
		loop
			int locationidx=1;
			bool cond;
			int Mouse_on_count = response_manager.total_response_count(1);
			int right_click_count = response_manager.total_response_count(3);
			int space_press = response_manager.total_response_count(4);
		until
			(cond && right_click_count<response_manager.total_response_count(3)) || space_press<response_manager.total_response_count(4)
		begin
			if response_manager.total_response_count(1)>Mouse_on_count then
				array<int> point_chosen[2];
				Mouse_on_count=response_manager.total_response_count(1);
				loop int i=1 until i>item_order.count() begin
					if mse.y()>image_pos_quiz[i][2]-pic_side_size/2 && mse.y()<image_pos_quiz[i][2]+pic_side_size/2 then
						if mse.x()>image_pos_quiz[i][1]+pic_side_size*2 && mse.x()<image_pos_quiz[i][1]+pic_side_size*3 then
							# no on item i
							answers[item_order[i]]=10;
							quiz.insert_part(locationidx,yellowbox,image_pos_quiz[i][1]+pic_side_size*2.5,image_pos_quiz[i][2]);
							quiz.insert_part(locationidx+1,whitebox,image_pos_quiz[i][1]+pic_side_size*4,image_pos_quiz[i][2]);
							locationidx=locationidx+2;
							NullTrial_woPort.get_stimulus_event(1).set_event_code("YES: Scene #"+string(scene)+", Item #"+string(item_order[i])+", Question:"+Questions[questionnum]);
							break;
						elseif mse.x()>image_pos_quiz[i][1]+pic_side_size*3.5 && mse.x()<image_pos_quiz[i][1]+pic_side_size*4.5 then
							# yes on item i
							answers[item_order[i]]=12;
							quiz.insert_part(locationidx,whitebox,image_pos_quiz[i][1]+pic_side_size*2.5,image_pos_quiz[i][2]);
							quiz.insert_part(locationidx+1,yellowbox,image_pos_quiz[i][1]+pic_side_size*4,image_pos_quiz[i][2]);
							locationidx=locationidx+2;
							NullTrial_woPort.get_stimulus_event(1).set_event_code("NO:  Scene #"+string(scene)+", Item #"+string(item_order[i])+", Question:"+Questions[questionnum]);
							break; 
						end;
					end;
					i=i+1;
				end;
				if arithmetic_mean(answers)>=10 && !cond then
					quiz.insert_part(quiz.part_count()-1,continue_text,0,-display_device.height()*5/12);
					right_click_count = response_manager.total_response_count(3);		
					cond=true;
				end;
			end;
			mse.poll();
			quiz.set_part_x(quiz.part_count(),mse.x());
			quiz.set_part_y(quiz.part_count(),mse.y());
			quiz.present();
		end;
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Quiz done, scene="+string(scene));
		NullTrial_wPort.present();
		loop int i=1 until i>answers.count() begin answers[i]=answers[i]-11;i=i+1;end;
		default.present();wait_interval(500);
		return answers;
	end;



#################################################################################################################################################################
######################################### RUNNING SCRIPT ########################################################################################################
#################################################################################################################################################################
	array<int>cond[num_scenes];
	
	int trialnum=1;
 #Show all locations on circle
#	loop int i=1 until i>temp_positions.count() begin
#		circle_pic[place_holder].add_part(circ2,temp_positions[i][1],temp_positions[i][2]);
#		i=i+1;
#	end;
#	circle_pic[place_holder].present();
#	loop int count = response_manager.total_response_count(3) until response_manager.total_response_count(3) > count begin end; wait_interval(5); # wait for the right mouse button press
#	loop until circle_pic[place_holder].part_count()==3 begin circle_pic[place_holder].remove_part(circle_pic[place_holder].part_count());end; #clean up image

 #Show all locations on circle, but set by set
#		loop int i=1 until i>true_positions.count() begin
#			loop until circle_pic[place_holder].part_count()==3 begin circle_pic[place_holder].remove_part(circle_pic[place_holder].part_count());end; #clean up image
#			loop int j=1 until j>6 begin
#				if true_positions[(i-1)*num_images_per_scene+j][1]!=0 && true_positions[(i-1)*num_images_per_scene+j][2]!=0 then
#					circle_pic[place_holder].add_part(circ2,true_positions[(i-1)*num_images_per_scene+j][1],true_positions[(i-1)*num_images_per_scene+j][2]);
#				end;
#				j=j+1;
#			end;
#			i=i+1;
#			circle_pic[place_holder].present();
#			loop int count3 = response_manager.total_response_count(3) until response_manager.total_response_count(3) > count3  begin end; wait_interval(5); # wait for the right mouse button press
#		end;
	
	if RunType!="Sleep" then # a task to test vigilance and "wake subjects up"
		if Initial_max_response_time==0 then
			Initial_max_response_time=450;
		end;
		if instructions("Welcome!\n\nBefore we start the main task, here is a short vigilance test.\nA red box will flicker on the screen.\n<b>When it pauses</b>, you have to determine whether it is on the right or left.\n\nIndicate the location using the left and right mouse buttons.\n\nYou must answer fast!\nAfter several correct trials, you will proceed to the main task.\n\nPress the right mouse button to start.",1) then
			stimulus_data last = stimulus_manager.last_stimulus_data();
			output_port port = output_port_manager.get_port(1);
			port.set_pulse_width(20);
			wait_interval(1000);
			array<int> correct[]={0,0,0,0,0,0,0,0,0,0};
			wakeuptask.set_duration(Initial_max_response_time);#defines the maximum required RT
			int INIT_MAX_REPS_FOR_VIGILANCE_TEST=30;
			int MAX_REPS_FOR_VIGILANCE_TEST=INIT_MAX_REPS_FOR_VIGILANCE_TEST;
			int i=1;
			int try=1;
			int count4=response_manager.total_response_count(4);
			loop until arithmetic_mean(correct)>=0.8 || response_manager.total_response_count(4)>count4 begin
				int delay_per_flicker=100;
				int num_flickers=random(10,30);
				loop int j=1 until j==num_flickers begin
					redboxpic.set_part_x(2,360*pow(-1,j));
					redboxpic.present();
					j=j+1;
					wait_interval(delay_per_flicker);
				end;
				port.send_code(1+int(mod(num_flickers,2)));#2 - right, 1 - left
				redbox.set_color(0,255,0);
				wakeuptaskStim.set_target_button(1+2*mod(num_flickers,2));
				wakeuptask.set_terminator_button(1+2*mod(num_flickers,2));
				wakeuptaskStim.set_event_code("Wake-up: Press "+string(1+2*mod(num_flickers,2)));
				int start_time=clock.time();
				wakeuptask.present();
				last = stimulus_manager.last_stimulus_data();
				if (last.type() == stimulus_hit) then
					redboxpic.present();
					correct[1+mod(i-1,10)]=1;
					term.print_line("Correct");
				else
					correct[1+mod(i-1,10)]=0;
					term.print_line("Miss/Incorrect");				
				end;
				redbox.set_color(255,0,0);					
				wait_interval(start_time+Initial_max_response_time-clock.time());
				i=i+1;
				if i>=MAX_REPS_FOR_VIGILANCE_TEST && (last.type() != stimulus_hit) then
					if !FileOpen && try>2 then
						loop until !instructions("Please call the experimenter into the room.",1) begin end;
					else
						if instructions("Let's try one more time.\n\nPress the right button to start.",1) then end;
						if FileOpen then
							Initial_max_response_time=Initial_max_response_time+50;
							wakeuptask.set_duration(Initial_max_response_time);
						end;
						i=1+mod(i-1,10);
						MAX_REPS_FOR_VIGILANCE_TEST=INIT_MAX_REPS_FOR_VIGILANCE_TEST+i-1;
					end;
					try=try+1;
				end;
			end;
			if FileOpen then
				out.print(Initial_max_response_time);out.print( "\n" );
			end;
		end;


		picture SSS=new picture;
		SSS.set_background_color(255,255,255);
		int box_height=100;
		int box_width=1700;
		box_height=int(box_height*scaling_factor);
		box_width=int(box_width*scaling_factor);
		array<double>unit_square3[][]={{-1.0,-1},{-1,1},{1,1},{1,-1}}; #Used to set the frame square around the image
		array<double>box_square3[4][2];
		loop int i=1 until i>4 begin box_square3[i][1]=unit_square3[i][1]*box_width/2; box_square3[i][2]=unit_square3[i][2]*box_height/2; i=i+1; end;
		loop int i=1 until i>7 begin
			text boxtext=new text;
			if i==1 then
				boxtext.set_caption("Feeling active, vital, alert, or wide awake");
			elseif i==2 then
				boxtext.set_caption("Functioning at high levels, but not at peak; able to concentrate");
			elseif i==3 then
				boxtext.set_caption("Awake, but relaxed; responsive but not fully alert");
			elseif i==4 then
				boxtext.set_caption("Somewhat foggy, let down");
			elseif i==5 then
				boxtext.set_caption("Foggy; losing interest in remaining awake; slowed down");
			elseif i==6 then
				boxtext.set_caption("Sleepy, woozy, fighting sleep; prefer to lie down");
			elseif i==7 then
				boxtext.set_caption("No longer fighting sleep, sleep onset soon; having dream-like thoughts");
			end;
			boxtext.set_font_size(int(40*scaling_factor));
			boxtext.set_background_color(255,255,255);
			boxtext.set_font_color(0,0,0);
			boxtext.redraw();
			line_graphic box_for_text=new line_graphic();
			box_for_text.set_line_color(0,0,0,255);
			box_for_text.set_fill_color(255,255,255,0);
			box_for_text.set_line_width(2);
			box_for_text.add_polygon(box_square3,true,1,0);
			box_for_text.redraw();
			SSS.add_part(boxtext,0,-display_device.height()/2+(8-i)*display_device.height()/9);
			SSS.add_part(box_for_text,0,-display_device.height()/2+(8-i)*display_device.height()/9);
			i=i+1;
		end;
		
		Instruct_text.set_caption("Good job!\nNow, please rate your level of sleepiness using the following scale:");
		Instruct_text.redraw();
		SSS.add_part(Instruct_text,0,display_device.height()*5/12);		

		SSS.add_part(MouseMarker,0,0);
		SSS.set_part_on_top(SSS.part_count(),true);
		mse.set_xy(0,0);

		NullTrial_wPort.get_stimulus_event(1).set_event_code("SSS");
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		screentype.set_port_code(2);
		SSS.present();
		NullTrial_woPort.present();
		screentype.set_port_code(0);
		
		int chosen_option;
		loop
			int Mouse_on_count = response_manager.total_response_count(1);
			bool first=true; # to mark first mouse motion
			int space_press = response_manager.total_response_count(4);
		until
			chosen_option!=0 || space_press<response_manager.total_response_count(4)
		begin
			if first && mse.x()!=0 && mse.y()!=0 then
				NullTrial_wPort.get_stimulus_event(1).set_event_code("Mouse Move");
				NullTrial_wPort.get_stimulus_event(1).set_port_code(8); #mouse move code
				NullTrial_wPort.present();
				first=false;
			end;
			if response_manager.total_response_count(1)>Mouse_on_count then
				int Mouse_off_count = response_manager.total_response_count(2);
				array<int> point_chosen[2];
				Mouse_on_count=response_manager.total_response_count(1);
				if mse.x()>-box_width/2 && mse.x()<box_width/2 then
					loop int i=1 until i>7 begin
						if mse.y()>-display_device.height()/2+(8-i)*display_device.height()/9-box_height/2 && mse.y()<-display_device.height()/2+(8-i)*display_device.height()/9+box_height/2 then
							chosen_option=i;
							break;
						end;
						i=i+1;
					end;
				end;
				
				loop until response_manager.total_response_count(2)>Mouse_off_count begin
					mse.poll();
					SSS.set_part_x(SSS.part_count(),mse.x());
					SSS.set_part_y(SSS.part_count(),mse.y());					
					SSS.present();
				end;
			end;
			mse.poll();
			SSS.set_part_x(SSS.part_count(),mse.x());
			SSS.set_part_y(SSS.part_count(),mse.y());
			SSS.present();
		end;
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		NullTrial_wPort.get_stimulus_event(1).set_event_code("SSS="+string(chosen_option));
		term.print_line("SSS="+string(chosen_option));
		NullTrial_wPort.present();
		default.present();wait_interval(500);
	end;		

	if RunType=="Training" then

		if instructions("Next, you will begin a task including multiple images,\npresented in sequence.\n\nThe images will include either a location, a face, or an abstract form.\n\nSome of the images will be repeated twice in a row,\nand your job will be to left-click when you see an image that was\npresented immediately before.\n\n\nThis part of the task will be approximately "+string(loc_duration/1000/60)+" minutes.\n\n\nPlease let the experimenter know if you have any questions.\nIf not, right-click when you are ready to start.",16) then
			string strsec=string(mod(loc_duration/1000,60));
			if strsec.count()==1 then
				strsec="0"+strsec;
			end;
			term.print_line("Total duration is "+string(loc_duration/1000/60)+":"+strsec+" minutes, start time is "+date_time("hh:nn:ss"));
			loc_trial.present();
			
			
			
			NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
			NullTrial_wPort.present();
		end;

/*		if instructions("TEMP",17) then
			trial loc_trial2=new trial();
			loc_trial2.set_type(specific_response);
			loc_trial2.set_terminator_button(4);
			loc_trial2.set_duration(1000*100*4);
			stimulus_event tmp_loc_stim_ev=loc_trial2.add_stimulus_event(loc_ITI);
			tmp_loc_stim_ev.set_event_code("Localization Start");
			tmp_loc_stim_ev.set_port_code(28);
			tmp_loc_stim_ev.set_duration(loc_ITI_max);
			array<int>loc_stims2[100];
			loc_stims2.fill(1,50,1,0);loc_stims2.fill(51,100,2,0);
			loc_stims2.shuffle();
			
			loop int i=1 until i>loc_stims2.count() begin
				sound tmpsnd=new sound(new wavefile("./Stim"+string(loc_stims2[i])+".wav"));
				tmpsnd.set_attenuation(0.2);
				tmp_loc_stim_ev=loc_trial2.add_stimulus_event(tmpsnd);
				tmp_loc_stim_ev.set_delta_time(loc_trial2.get_stimulus_event(loc_trial2.stimulus_event_count()-1).duration());
				if loc_stims2[i]==1 then
					tmp_loc_stim_ev.set_event_code("Scene , trial #"+string(i));
					tmp_loc_stim_ev.set_port_code(21);
				else
					tmp_loc_stim_ev.set_event_code("Object, trial #"+string(i));
					tmp_loc_stim_ev.set_port_code(22);
				end;
				tmp_loc_stim_ev.set_duration(loc_present_time);
				tmp_loc_stim_ev=loc_trial2.add_stimulus_event(loc_ITI);
				tmp_loc_stim_ev.set_delta_time(loc_trial2.get_stimulus_event(loc_trial2.stimulus_event_count()-1).duration());
				tmp_loc_stim_ev.set_event_code("Localization ITI");
				tmp_loc_stim_ev.set_port_code(28);
				tmp_loc_stim_ev.set_duration(random(loc_ITI_min,loc_ITI_max)+1000);
				i=i+1;
			end;
			loc_trial2.present();
			
			NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
			NullTrial_wPort.present();
		end;
*/
		vid0.prepare();
		set_system_volume(0.5,1);
		vidpresent0.present();
		set_system_volume(1,1);
		vid0.release();
		
		loop until !instructions("Any questions?",3) begin end;

		vid1.prepare();
		set_system_volume(0.5,1);
		vidpresent1.present();
		set_system_volume(1,1);
		vid1.release();		
		
		#loop until !instructions("Wait for instructions",3) begin end;
		loop until !instructions("Any questions?",3) begin end;
		#if instructions("Right click to show first set of images",4) then
		if instructions("<b><u>Additional note:</u></b>\nYou will be asked a series of yes/no questions\nabout your story after each recording.\nCarefully select the best answer to each question.\n\nIf the answer is not a clear-cut yes or no\nchoose the option that makes more sense.\n\n\n\nAny Questions?\n\n\n\nIf not, right click to show the first set of images",4) then
			array<int>rand_ord[num_scenes];rand_ord.fill(1,rand_ord.count(),1,1);rand_ord.shuffle();
			
			loop int j=1 until j>objects.count() begin
				objects[j].set_size(double(pic_side_size*3),double(pic_side_size*3),0.0);
				j=j+1;
			end;
			loop int j=1 until j>num_scenes begin
				term.print("Scene number "+string(rand_ord[j])+": "+string(j)+"/"+string(num_scenes)+" - ");
				#exposure			
				no_circle_pic[rand_ord[j]].add_part(Background,display_device.width()/2-Radius,0);
				no_circle_pic[rand_ord[j]].add_part(Instruct_text,-Radius-25,display_device.height()/2-40+y_shift_left/2);
				#no_circle_pic[rand_ord[j]].add_part(Instruct_text,350,2.05*display_device.height()/5);	
				Instruct_text.set_background_color(255,255,255);
				Instruct_text.set_caption("Do not speak just yet.\nRight click to record your story."); Instruct_text.redraw();
				array<int>item_order[num_images_per_scene];item_order.fill(1,item_order.count(),1,1);item_order.shuffle();
				array<double>image_pos[4][2]={{150.0,-250},{550,-250},{150,150},{550,150}};

#				image_pos[1][1]=double(display_device.width()/4)-200;				image_pos[1][2]=-200.0;
#				image_pos[2][1]=double(display_device.width()/4)+200;				image_pos[2][2]=-200.0;
#				image_pos[3][1]=double(display_device.width()/4)-200;				image_pos[3][2]=200.0;
#				image_pos[4][1]=double(display_device.width()/4)+200;				image_pos[4][2]=200.0;
				
				array<int>image_left_pos[4][2]={{-585,-300},{-545,-300},{-585,-280},{-545,-280}};
				array<bitmap> imgs_left[num_images_per_scene];
				
				item_frame_large.set_line_color(colors[rand_ord[j]][1],colors[rand_ord[j]][2],colors[rand_ord[j]][3],255);	item_frame_large.redraw();
				item_frame_left.set_line_color(colors[rand_ord[j]][1],colors[rand_ord[j]][2],colors[rand_ord[j]][3],255);	item_frame_left.redraw();
				if rand_ord[j]==3 then
					item_frame_large.set_line_color(0,0,0,255);	item_frame_large.redraw();
					item_frame_left.set_line_color(0,0,0,255);	item_frame_left.redraw();
				end;		
				loop int n=1 until n>item_order.count() begin
					#scene_object_and_sound.add_3dpart(objects[item_scene_allocation[(rand_ord[j]-1)*num_images_per_scene+item_order[n]]],double(n*display_device.width()/(num_images_per_scene+1)-display_device.width()/2),-275.0,0.0);	
					imgs_left[n]=new bitmap; imgs_left[n].set_load_size(double(pic_side_size)/10,double(pic_side_size)/4,0);
					imgs_left[n].set_filename(objects[item_scene_allocation[(rand_ord[j]-1)*num_images_per_scene+item_order[n]]].get_texture(1).filename());
					imgs_left[n].load();
					no_circle_pic[rand_ord[j]].add_3dpart(objects[item_scene_allocation[(rand_ord[j]-1)*num_images_per_scene+item_order[n]]],image_pos[n][1],image_pos[n][2],0.0);	
					no_circle_pic[rand_ord[j]].add_part(imgs_left[n],image_left_pos[n][1],image_left_pos[n][2]+y_shift_left);	
					no_circle_pic[rand_ord[j]].add_part(item_frame_large,int(image_pos[n][1]),int(image_pos[n][2]));	
					no_circle_pic[rand_ord[j]].set_part_on_top(no_circle_pic[rand_ord[j]].part_count(),true);
					no_circle_pic[rand_ord[j]].add_part(item_frame_left,image_left_pos[n][1],image_left_pos[n][2]+y_shift_left);	
					text tmptext = new text();	tmptext.set_background_color(255,255,255,0);tmptext.set_font_color(0,0,0);	tmptext.set_align(align_center);			
					tmptext.set_caption(objectnames[item_scene_allocation[(rand_ord[j]-1)*num_images_per_scene+item_order[n]]]);
					tmptext.set_font_size(60);tmptext.redraw();
					if n<=2 then
						no_circle_pic[rand_ord[j]].add_part(tmptext,image_pos[n][1],image_pos[n][2]-245);	
					else
						no_circle_pic[rand_ord[j]].add_part(tmptext,image_pos[n][1],image_pos[n][2]+245);	
					end;
					n=n+1;
				end;
				status_txt.set_caption("Scene #"+string(j)+"\nout of "+string(num_scenes));status_txt.redraw();
				no_circle_pic[rand_ord[j]].add_part(status_txt,display_device.width()/2-100,-display_device.height()/2+50);
				NullTrial_wPort.get_stimulus_event(1).set_port_code(rand_ord[j]);
				NullTrial_wPort.get_stimulus_event(1).set_event_code("ExposureObjects: Scene #"+string(rand_ord[j]));
				no_circle_pic[rand_ord[j]].present();
				NullTrial_wPort.present();

				loop int n=1;int partcount=no_circle_pic[rand_ord[j]].part_count() until n>item_order.count() begin
					no_circle_pic[rand_ord[j]].remove_part(partcount-n*4+2);
					no_circle_pic[rand_ord[j]].remove_part(partcount-n*4);
					n=n+1;
				end;
				no_circle_pic[rand_ord[j]].remove_part(1);
				wait_interval(2500);
				int num_steps=30;
				double step_size=(250+y_shift_left-(-50))/num_steps;
				loop int i=1 until i>num_steps begin
					no_circle_pic[rand_ord[j]].set_part_y(1,250+y_shift_left-int(i*step_size));
					no_circle_pic[rand_ord[j]].set_part_y(2,250+y_shift_left-int(i*step_size));
					no_circle_pic[rand_ord[j]].present();
					i=i+1;
				end;				
				no_circle_pic[rand_ord[j]].set_part_y(1,250+y_shift_left);
				no_circle_pic[rand_ord[j]].set_part_y(2,250+y_shift_left);
				no_circle_pic[rand_ord[j]].insert_part(1,roomimage,-Radius-25,y_shift_left);

				NullTrial_wPort.get_stimulus_event(1).set_port_code(27); NullTrial_wPort.present();
				loop int n=1 until n>item_order.count() begin
					no_circle_pic[rand_ord[j]].remove_3dpart(no_circle_pic[rand_ord[j]].d3d_part_count());
					n=n+1;
				end;
				loop int n=1 until n>item_order.count()*2+3 begin
					no_circle_pic[rand_ord[j]].remove_part(no_circle_pic[rand_ord[j]].part_count());
					n=n+1;
				end;
				int count4 = response_manager.total_response_count(4);
				sound_recording rec=new sound_recording();
				rec.set_duration(600000);
				
				string filename=tempsoundforfilename.filename().replace("stim\\1","log\\S"+SubjectNum+"_"+string(rand_ord[j])+"_1");
				loop int i=2 until !file_exists(filename) begin		filename=filename.replace(string(i-1)+".wav",string(i)+".wav");i=i+1;	end;				
#$$$#				filename=filename.substring(filename.find("log\\S")+4,filename.count()-(filename.find("log\\S")+3));
				rec.set_base_filename(filename);
				rec.set_use_counter(false);
				rec.set_use_date_time(false);
				trial recording=new trial();
				recording.set_duration(600000);
				recording.add_stimulus_event(rec);
				recording.set_type(specific_response);
				recording.set_terminator_button(3);
				cursor.set_caption("Please briefly describe the story.\n\n\nRight click when done.");cursor.redraw();
				loop int count3 = response_manager.total_response_count(3) until response_manager.total_response_count(3)>count3 || response_manager.total_response_count(4)>count4 begin end;wait_interval(20); # wait for the right mouse button or space
				default.present();
				int start_time=clock.time();
				recording.present();
				recording_manager.stop_all();
				term.print_line(string(double(clock.time()-start_time)/1000)+" seconds");				
				cursor.set_caption("<b>+</b>");cursor.redraw();
				default.present();
				wait_interval(1000);
				array<int>question_ord[Questions.count()];

				question_ord.fill(1,question_ord.count(),1,1);#question_ord.shuffle();
				loop int i=1 until i>question_ord.count() begin
					Answers[rand_ord[j]][question_ord[i]]=record_answers(rand_ord[j],question_ord[i]);
					i=i+1;
				end;
				j=j+1;
			end;				
			status_txt.set_caption(" ");status_txt.redraw();
			loop int j=1 until j>objects.count() begin
				objects[j].set_size(double(pic_side_size),double(pic_side_size),0.0);
				j=j+1;
			end;			
			output_file outAnswers = new output_file;
			outAnswers.open(".\\Answers_"+SubjectNum+".txt");
			loop int i=1 until i>num_scenes begin
				loop int j=1 until j>Questions.count() begin
					loop int n=1 until n>num_images_per_scene begin
						outAnswers.print(Answers[i][j][n]);outAnswers.print( "\n" );						
						n=n+1;
					end;
					j=j+1;
				end;
				i=i+1;
			end;
			outAnswers.close();	
		end;

		trialnum=1;
		array<bool>was_previous_response_correct[true_positions.count()];
		vid2.prepare();
		set_system_volume(0.5,1);
		vidpresent2.present();
		set_system_volume(1,1);
		vid2.release();
		
		loop until !instructions("Any questions?",5) begin end;
		if instructions("Right-click to start a practice round with just three tiles.",6) then
			array<int>stimorder[3];
			stimorder.fill(1,stimorder.count(),num_scenes*num_images_per_scene+1,1);stimorder.shuffle();
			term.print_line("Stage\tBlock\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename");
			term.print_line("-----\t-----\t---\t-------\t------\t--------\t--------\t--------\t--------\t----\t-----------\t--------");
			loop int i=1 until i>stimorder.count() begin
				out_results.print("2\t0\t0\t"+string(stimorder[i])+"\t\t"+string(true_positions[stimorder[i]][1])+"\t"+string(true_positions[stimorder[i]][2])+"\t\t\t\t0\t"+objects[item_scene_allocation[stimorder[i]]].get_texture(1).filename()+"\n");
				term.print_line("2\t0\t0\t"+string(stimorder[i])+"\t\t\t"+string(true_positions[stimorder[i]][1])+"\t\t"+string(true_positions[stimorder[i]][2])+"\t\t\t\t\t\t\t0\t\t"+objects[item_scene_allocation[stimorder[i]]].get_texture(1).filename());
				exposure_only(stimorder[i]);
				i=i+1;
				trialnum=trialnum+1;
			end;
			stimorder.shuffle();
			int repnum=1;
			status_txt.set_caption("0/"+string(num_practice_images)+"\nlearned to\ncriterion");status_txt.redraw();
			loop int i=1 until i>stimorder.count() begin
				if mod(trialnum,47)==0 then
					term.print_line("Stage\tBlock\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename");
					term.print_line("-----\t-----\t---\t-------\t------\t--------\t--------\t--------\t--------\t----\t-----------\t--------");
				end;
				int last_presented;
				out_results.print("2\t0\t0\t"+string(stimorder[i])+"\t"+string(repnum)+"\t"+string(true_positions[stimorder[i]][1])+"\t"+string(true_positions[stimorder[i]][2])+"\t");
				term.print("2\t0\t0\t"+string(stimorder[i])+"\t\t"+string(repnum)+"\t"+string(true_positions[stimorder[i]][1])+"\t\t"+string(true_positions[stimorder[i]][2])+"\t\t");
				array<int>temp_pos[3]=positioning_only(stimorder[i],true,true,false,was_previous_response_correct[stimorder[i]]);			
				double dist=temp_pos[3];
				temp_pos.resize(2);
				last_presented=stimorder[i];
				if dist<=correct_incorrect_criterion && was_previous_response_correct[stimorder[i]] then
					array<int> temparray[stimorder.count()-1];
					loop int j=1; int idx2=1 until j>stimorder.count() begin
						if j!=i then
							temparray[idx2]=stimorder[j];
							idx2=idx2+1;
						end;
						j=j+1;
					end;
					stimorder.assign(temparray);
					status_txt.set_caption(string(num_practice_images-stimorder.count())+"/"+string(num_practice_images)+"\nlearned to\ncriterion");status_txt.redraw();					
				elseif dist<=correct_incorrect_criterion then
					was_previous_response_correct[stimorder[i]]=true;
					i=i+1;
				else
					was_previous_response_correct[stimorder[i]]=false;
					i=i+1;
				end;
				if i>stimorder.count() then
					repnum=repnum+1;
					if stimorder.count()>1 then
						stimorder.shuffle();
						loop until stimorder[1]!=last_presented begin
							stimorder.shuffle();					
						end;
					end;
					i=1;
				end;
				trialnum=trialnum+1;			
			end;			
			status_txt.set_caption(" ");status_txt.redraw();				
		end;

		vid3.prepare();
		set_system_volume(0.5,1);
		vidpresent3.present();
		set_system_volume(1,1);
		vid3.release();

		#if instructions("Good job!\n\nWait for instructions\n\n\n\n\n\n\n\n\n\n\n\n",7) then
		loop until !instructions("Any questions?",7) begin end;

		vid4.prepare();
		set_system_volume(0.5,1);
		vidpresent4.present();
		set_system_volume(1,1);
		vid4.release();
#			loop until !instructions_with_image("\n\n\n\n\n\n\n\n\n\n\n\n\n\n",8) begin end;
		#end;
		loop until !instructions("Any questions?",8) begin end;
		vid5.prepare();
		set_system_volume(0.5,1);
		vidpresent5.present();
		set_system_volume(1,1);
		vid5.release();
		loop until !instructions("Any questions?",8) begin end;
		
		#instructions_with_image("The task will consist of "+string(num_scenes/num_scenes_per_block)+" blocks, each including "+string(num_scenes_per_block*num_images_per_scene)+" objects.\nAfter these blocks, a final test will check your memory on all "+string(num_scenes*num_images_per_scene)+" objects.\nA similar test will be run after the nap.\nBased on your scores in both tests, you may recieve a bonus payment of up to 16$!\nTry to be as exact as possible in the object placement to get more money!\n\n\n\n\n\n\n\n\n\n\nGood Luck!\nPress the right button to begin the first block.",10);

		loop int blocknum=1 until blocknum>scenes_per_block.count()/num_scenes_per_block begin
			array<int>stimorder[num_scenes_per_block*num_images_per_scene];
			array<int>scenes_in_this_block[0];
			loop int i=1 until i>num_scenes_per_block begin
				stimorder.fill((i-1)*num_images_per_scene+1,(i-1)*num_images_per_scene+4,(scenes_per_block[(blocknum-1)*num_scenes_per_block+i]-1)*num_images_per_scene+1,1);
				scenes_in_this_block.add(scenes_per_block[(blocknum-1)*num_scenes_per_block+i]);
				i=i+1;
			end;
			scenes_in_this_block.shuffle();
			term.print_line(stimorder);
			stimorder=shuffle_with_no_consec(stimorder);
			term.print_line("Stage\tBlock\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename");
			term.print_line("-----\t-----\t---\t-------\t------\t--------\t--------\t--------\t--------\t----\t-----------\t--------");
			term.print("Rep #0 starting; Stimulus order:");
			term.print_line(stimorder);
			if hear_sounds_stories(scenes_in_this_block,blocknum) then
#instructions("We will now start <u>Block #"+string(blocknum)+"</u> out of "+string(scenes_per_block.count()/num_scenes_per_block)+"\nof the spatial task.\n\nGood luck!\n\nRight-click to proceed.",9+blocknum*3-2) then		
				# EXPOSURE
				loop int i=1 until i>stimorder.count() begin
					out_results.print("2\t"+string(blocknum)+"\t"+string((stimorder[i]-1)/num_images_per_scene+1)+"\t"+string(mod(stimorder[i]-1,num_images_per_scene)+1)+"\t0\t"+string(true_positions[stimorder[i]][1])+"\t"+string(true_positions[stimorder[i]][2])+"\t\t\t\t1\t"+objects[item_scene_allocation[stimorder[i]]].get_texture(1).filename()+"\n");
					term.print_line("2\t"+string(blocknum)+"\t"+string((stimorder[i]-1)/num_images_per_scene+1)+"\t"+string(mod(stimorder[i]-1,num_images_per_scene)+1)+"\t\t0\t"+string(true_positions[stimorder[i]][1])+"\t\t"+string(true_positions[stimorder[i]][2])+"\t\t\t\t\t\t\t1\t\t"+objects[item_scene_allocation[stimorder[i]]].get_texture(1).filename());
					exposure_only(stimorder[i]);
					i=i+1;
					trialnum=trialnum+1;
				end;
			end;				
			stimorder=shuffle_with_no_consec(stimorder);
			term.print("Rep #1 starting; Stimulus order:");
			term.print_line(stimorder);
			if instructions("<u>Block #"+string(blocknum)+"</u>\n\n\nYou will now be tested on these image-locations.\n\n\nRight-click to proceed.",10) then		
				# POSITIONING
				int repnum=1;
				status_txt.set_caption("0/"+string(num_scenes_per_block*num_images_per_scene)+"\nlearned to\ncriterion");status_txt.redraw();
				loop int i=1 until i>stimorder.count() begin
					if mod(trialnum,47)==0 then
						term.print_line("Stage\tBlock\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename");
						term.print_line("-----\t-----\t---\t-------\t------\t--------\t--------\t--------\t--------\t----\t-----------\t--------");
					end;
					int last_presented;
								#Showing the real location of the screen
								#loop until circle_pic[place_holder].part_count()==3 begin circle_pic[place_holder].remove_part(circle_pic[place_holder].part_count());end; #clean up image
								#circle_pic[place_holder].add_part(circ2,true_positions[blocks[blocknum][stimorder[i]][1]][blocks[blocknum][stimorder[i]][2]][1],true_positions[blocks[blocknum][stimorder[i]][1]][blocks[blocknum][stimorder[i]][2]][2]);

					out_results.print("2\t"+string(blocknum)+"\t"+string((stimorder[i]-1)/num_images_per_scene+1)+"\t"+string(mod(stimorder[i]-1,num_images_per_scene)+1)+"\t"+string(repnum)+"\t"+string(true_positions[stimorder[i]][1])+"\t"+string(true_positions[stimorder[i]][2])+"\t");
					term.print("2\t"+string(blocknum)+"\t"+string((stimorder[i]-1)/num_images_per_scene+1)+"\t"+string(mod(stimorder[i]-1,num_images_per_scene)+1)+"\t\t"+string(repnum)+"\t"+string(true_positions[stimorder[i]][1])+"\t\t"+string(true_positions[stimorder[i]][2])+"\t\t");
					#out_results.print("3\t"+string(blocknum)+"\t"+string(stimorder[i])+"\t"+string(order_of_presentation[stimorder[i]][1][blocknum])+"\t"+string(repnum)+"\t"+string(true_positions[stimorder[i]][order_of_presentation[stimorder[i]][1][blocknum]][1])+"\t"+string(true_positions[stimorder[i]][order_of_presentation[stimorder[i]][1][blocknum]][2])+"\t");
					#term.print("3\t"+string(blocknum)+"\t"+string(stimorder[i])+"\t"+string(order_of_presentation[stimorder[i]][1][blocknum])+"\t\t"+string(repnum)+"\t"+string(true_positions[stimorder[i]][order_of_presentation[stimorder[i]][1][blocknum]][1])+"\t\t"+string(true_positions[stimorder[i]][order_of_presentation[stimorder[i]][1][blocknum]][2])+"\t\t");
					array<int>temp_pos[3]=positioning_only(stimorder[i],true,true,true,was_previous_response_correct[stimorder[i]]);					
					double dist=temp_pos[3];
					temp_pos.resize(2);
					last_presented=stimorder[i];
					if dist!=-1 && dist<=correct_incorrect_criterion && was_previous_response_correct[stimorder[i]] then
						array<int> temparray[stimorder.count()-1];
						loop int j=1; int idx2=1 until j>stimorder.count() begin
							if j!=i then
								temparray[idx2]=stimorder[j];
								idx2=idx2+1;
							end;
							j=j+1;
						end;
						stimorder.assign(temparray);
						status_txt.set_caption(string(num_scenes_per_block*num_images_per_scene-stimorder.count())+"/"+string(num_scenes_per_block*num_images_per_scene)+"\nlearned to\ncriterion");status_txt.redraw();
					elseif dist!=-1 && dist<=correct_incorrect_criterion then
						was_previous_response_correct[stimorder[i]]=true;
						i=i+1;
					elseif dist!=-1 then
						was_previous_response_correct[stimorder[i]]=false;
						i=i+1;
					else
						i=i+1;
					end;
					if i>stimorder.count() then
						repnum=repnum+1;
						if stimorder.count()>1 then
							stimorder=shuffle_with_no_consec(stimorder);
							array<int> scenes_left_in_block[0];scenes_left_in_block.add((stimorder[1]-1)/num_images_per_scene);
							loop int j=2 until j>stimorder.count() begin
								if (stimorder[j]-1)/num_images_per_scene!=scenes_left_in_block[1] then
									scenes_left_in_block.add((stimorder[j]-1)/num_images_per_scene);
									break;
								end;
								j=j+1;
							end;								
							loop int numtrys=1 until scenes_left_in_block.count()==1 || (stimorder[1]-1)/num_images_per_scene!=(last_presented-1)/num_images_per_scene || numtrys>100 begin
								stimorder=shuffle_with_no_consec(stimorder);
								numtrys=numtrys+1;
							end;
							loop until stimorder[1]!=last_presented begin
								stimorder=shuffle_with_no_consec(stimorder);
							end;
								
						end;
						term.print("Rep #"+string(repnum)+" starting; Stimulus order:");
						term.print_line(stimorder);
						i=1;
					end;
					trialnum=trialnum+1;			
				end;
				status_txt.set_caption(" ");status_txt.redraw();				
			end;
			
			blocknum=blocknum+1;
		end;

	end;
	
	int stagenum;
	string txt;
	if RunType=="Training" then
		#txt="Finally, you will be tested on the positions of all the object images.\nThis is the very last test before the end of this part.\n\nPlease try your very best, because this is the definitive test.\nYour bonus for today depends on the results of this test and the one after the nap.\n\n\n\n\n\n\n\n\n\n\nGood Luck!\nPress the right button to begin the test.";
		vid6.prepare();
		set_system_volume(0.5,1);
		vidpresent6.present();
		set_system_volume(1,1);
		vid6.release();
		loop until !instructions_with_image("Any questions?\n\n\n\n\n\n\n\n\n\n\n\n\n\n",7) begin end;
		txt="\n\nRight click to begin final test\n\n\n\n\n\n\n\n\n\n\n\n";
		stagenum=4;
	else
		txt="Welcome back!\You will now be tested on the positions of all the object images.\n\nPlease try your very best, because this is the definitive test.\nYour bonus for today depends on the results of this test and the one before the nap.\n\n\n\n\n\n\n\n\n\n\n\nGood Luck!\nPress the right button to begin the test.";
		stagenum=5;
	end;
	#string txt=" ddd";int stagenum=1;sounds[1].present();
	if RunType=="Training" || RunType=="T2" then		
		array<int>mean_error_per_item[num_scenes*num_images_per_scene];
		if instructions_with_image(txt,11) then		
			array<int>testorder[num_scenes*num_images_per_scene+3];
			testorder.fill(1,3,num_scenes*num_images_per_scene+1,1);testorder.fill(4,testorder.count(),1,1);
			loop until check_pseudorandom(testorder) begin testorder.shuffle(4,testorder.count()); end;
			term.print_line(testorder);
			# POSITIONING
			#Nothing_for_positioning.set_delta_time(0);
			#positioning_trial.set_duration(50);
			term.print_line("Stage\tTrial\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename");
			term.print_line("-----\t-----\t---\t-------\t------\t--------\t--------\t--------\t--------\t----\t-----------\t--------");
			loop int i=1 until i>testorder.count() begin
				if mod(i,47)==0 then
					term.print_line("Stage\tTrial\tSet\tPicture\tRepNum\tTrueLocX\tTrueLocY\tUserLocX\tUserLocY\tDiff\tSnd(1=Orig)\tFilename");
					term.print_line("-----\t-----\t---\t-------\t------\t--------\t--------\t--------\t--------\t----\t-----------\t--------");
				end;
				out_results.print(string(stagenum)+"\t0\t"+string((testorder[i]-1)/num_images_per_scene+1)+"\t"+string(mod(testorder[i]-1,num_images_per_scene)+1)+"\t1\t"+string(true_positions[testorder[i]][1])+"\t"+string(true_positions[testorder[i]][2])+"\t");
				term.print(string(stagenum)+"\t"+string(i)+"/"+string(testorder.count())+"\t"+string((testorder[i]-1)/num_images_per_scene+1)+"\t"+string(mod(testorder[i]-1,num_images_per_scene)+1)+"\t\t1\t"+string(true_positions[testorder[i]][1])+"\t\t"+string(true_positions[testorder[i]][2])+"\t\t");
				array<int>temp_pos[3]=positioning_only(testorder[i],false,true,false,false);
			
				double dist=temp_pos[3];
				temp_pos.resize(2);
				if i>3 then
					mean_error_per_item[testorder[i]]=int(dist);
					if dist<=correct_incorrect_criterion then
						sum_bonus=sum_bonus+(8.0/(num_scenes*num_images_per_scene))*double(correct_incorrect_criterion-int(dist))/correct_incorrect_criterion;
					end;
				end;
				i=i+1;
			end;
		else
			mean_error_per_item.fill(1,mean_error_per_item.count(),5,5);mean_error_per_item.shuffle();#for testing only
			term.print_line(mean_error_per_item);
		end;

		if RunType=="Training" then 			
			Instruct_text.set_caption("Thank you!\n\n\nWe're done with this part!");Instruct_text.redraw();
			NullTrial_wPort.get_stimulus_event(1).set_event_code("Thank you!\n\n\nWe're done with this part!");
			NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
			screentype.set_port_code(12);
			InstructPicWithoutImage.present();NullTrial_wPort.present();
			#choose TMRed
			array<int>cued_sound[0];
			term.print_line(mean_error_per_item);
/*
			# step 1 - find balanced couples within each scene
			array<int>permute_four[][]={{1,2,3,4},{1,3,2,4},{1,4,2,3},{2,3,1,4},{2,4,1,3},{3,4,1,2}};
			array<int>permute_per_scene[num_scenes];
			array<double>means[num_scenes][3];
			loop int i=1 until i>num_scenes begin
				array<int>scene_related_items[num_images_per_scene];
				loop int j=1 until j>scene_related_items.count() begin
					scene_related_items[j]=mean_error_per_item[num_images_per_scene*(i-1)+j];
					j=j+1;
				end;
				int min_dist=10000000;
				array<int>min_pos[0];
				loop int j=1 until j>permute_four.count() begin
					int dist=abs((scene_related_items[permute_four[j][1]]+scene_related_items[permute_four[j][2]])-(scene_related_items[permute_four[j][3]]+scene_related_items[permute_four[j][4]]));
					if dist<min_dist then
						min_dist=dist;
						min_pos.resize(0);
						min_pos.add(j);
					elseif dist==min_dist then
						min_pos.add(j);
					end;
					j=j+1;
				end;
				min_pos.shuffle();
				permute_per_scene[i]=min_pos[1];
				means[i][1]=arithmetic_mean(scene_related_items);
				means[i][2]=(double(scene_related_items[permute_four[permute_per_scene[i]][1]])+double(scene_related_items[permute_four[permute_per_scene[i]][2]]))/2;
				means[i][3]=(double(scene_related_items[permute_four[permute_per_scene[i]][3]])+double(scene_related_items[permute_four[permute_per_scene[i]][4]]))/2;
				i=i+1;
			end;
			input_file perm_in = new input_file;
			perm_in.open(".\\stim\\perms_"+string(num_scenes)+".txt");
			perm_in.set_delimiter( '\n' );
			array<int>perms[38760][6];
			if num_scenes==18 then
				perms.resize(18564);
			end;
			
			loop int i=1 until i>perms.count() begin
				loop int j=1 until j>perms[1].count() begin
					perms[i][j]=perm_in.get_int();
					j=j+1;
				end;
				i=i+1;
			end;
			perm_in.close();
			array<int>scenes_per_group[]={6,7,7};
			if num_scenes==18 then
				scenes_per_group={6,6,6};
			end;
			double min_var=10000000;
			array<int>min_pos[0];
			loop int j=1 until j>perms.count() begin
				array<double>three_scores[3];
				loop int i=1 until i>num_scenes begin
					bool cond2=false;
					loop int k=1 until k>scenes_per_group[1] begin if i==perms[j][k] then cond2=true; break; end; k=k+1; end;
					if cond2 then
						three_scores[1]=three_scores[1]+means[i][1]/scenes_per_group[1];
					else
						three_scores[2]=three_scores[2]+means[i][2]/(scenes_per_group[2]*2);
						three_scores[3]=three_scores[3]+means[i][3]/(scenes_per_group[3]*2);
					end;
					i=i+1;
				end;
				double var=(pow(three_scores[1]-arithmetic_mean(three_scores),2)+pow(three_scores[2]-arithmetic_mean(three_scores),2)+pow(three_scores[3]-arithmetic_mean(three_scores),2))/3;
				if var<min_var then
					min_var=var;
					term.print_line(var);term.print_line(j);
					min_pos.resize(0);
					min_pos.add(j);
				elseif var==min_var then
					term.print_line(var);term.print_line(j);
					min_pos.add(j);
				end;
				j=j+1;
			end;
			min_pos.shuffle();
			loop int i=1 until i>num_scenes begin
				bool cond2=true;
				loop int k=1 until k>scenes_per_group[1] begin if i==perms[min_pos[1]][k] then cond2=false; break; end; k=k+1; end;
				if cond2 then
					cued_sound.add((i-1)*num_images_per_scene+permute_four[permute_per_scene[i]][1]);
					cued_sound.add((i-1)*num_images_per_scene+permute_four[permute_per_scene[i]][2]);
				end;
				i=i+1;
			end;
*/
			array<int>groups[num_scenes*num_images_per_scene];
			groups.fill(1,6*num_images_per_scene ,1,0);
			loop int i=1 until i>6 begin
				groups.fill((4+2*i)*num_images_per_scene+1,(4+2*i)*num_images_per_scene+2,2,0);
				groups.fill((4+2*i)*num_images_per_scene+3,(5+2*i)*num_images_per_scene,3,0);
				groups.fill((5+2*i)*num_images_per_scene+1,(6+2*i)*num_images_per_scene,4,0);
				i=i+1;
			end;
			array<int>numel_in_groups[]={24,12,12,24};
			double min_var=100000000;
			array<int>min_pos[0];
			loop int n=1 until n>200000 begin
				array<int>rand_order_item[0];
				array<int>blockorder[scenes_per_block.count()/num_scenes_per_block];
				blockorder.fill(1,blockorder.count(),1,1);blockorder.shuffle();
				loop int i=1 until i>blockorder.count() begin
					int randnum=random(0,1);
					array<int>randord[num_images_per_scene];
					randord.fill(1,randord.count(),1,1);
					randord.shuffle();
					loop int j=1 until j>num_images_per_scene begin
						rand_order_item.add((scenes_per_block[blockorder[i]*2-randnum]-1)*num_images_per_scene+randord[j]);
						j=j+1;
					end;
					randord.shuffle();
					loop int j=1 until j>num_images_per_scene begin
						rand_order_item.add((scenes_per_block[blockorder[i]*2-(1-randnum)]-1)*num_images_per_scene+randord[j]);
						j=j+1;
					end;
					i=i+1;
				end;
				array<double>means[4];
				loop int i=1 until i>groups.count() begin
					means[groups[i]]=means[groups[i]]+double(mean_error_per_item[rand_order_item[i]])/numel_in_groups[groups[i]];
					i=i+1;
				end;
				double var=(pow(means[1]-arithmetic_mean(means),2)+pow(means[2]-arithmetic_mean(means),2)+pow(means[3]-arithmetic_mean(means),2)+pow(means[4]-arithmetic_mean(means),2))/4;
				if var<min_var then
					min_var=var;
					min_pos.resize(0);
					min_pos.append(rand_order_item);
					min_pos.add(n);
				end;
				n=n+1;
			end;
			# showing stats for final choice
			array<double>means[4];
			loop int i=1 until i>groups.count() begin
				means[groups[i]]=means[groups[i]]+double(mean_error_per_item[min_pos[i]])/numel_in_groups[groups[i]];
				i=i+1;
			end;
			double var=(pow(means[1]-arithmetic_mean(means),2)+pow(means[2]-arithmetic_mean(means),2)+pow(means[3]-arithmetic_mean(means),2)+pow(means[4]-arithmetic_mean(means),2))/4;
			term.print_line(means);				
			term.print_line(var);term.print_line(min_pos[min_pos.count()]);
			min_pos.resize(min_pos.count()-1);
			term.print_line(min_pos);
			
			loop int i=1 until i>min_pos.count() begin
				if groups[i]==2 then
					cued_sound.add(min_pos[i]);
				end;
				i=i+1;
			end;
			term.print_line(cued_sound);
			
			output_file out_for_TMR = new output_file;	
			string filename=TMRcues[1].get_wavefile().filename();
			filename=filename.replace("stim\\Stim0.wav","log\\TMR_list_for_"+SubjectNum+".txt");
			out_for_TMR.open(filename);
			loop int i=1 until i>cued_sound.count() begin
				out_for_TMR.print(string(cued_sound[i])+"\n");
				i=i+1;
			end;
			out_for_TMR.close();	
			term.print_line("Total bonus for part 1 was "+string(sum_bonus)+"$");
			NullTrial_woPort.get_stimulus_event(1).set_event_code("Total bonus for part 1 was "+string(sum_bonus)+"$");
			NullTrial_woPort.present();
		end;
		out_results.close();		

		if FileOpen then
			out.print(sum_bonus);
			out.close();
		end;
	end;


	if RunType=="T2" then
		term.print_line("Total bonus for both parts was "+string(sum_bonus)+"$");
		NullTrial_woPort.get_stimulus_event(1).set_event_code("Total bonus for both parts was "+string(sum_bonus)+"$");
		NullTrial_woPort.present();

		string temptext=Instruct_text.caption();
		Instruct_text.set_caption("Thanks for trying!\n\nBefore we're done, we have just one more test left.\n\nPlease grab the keyboard from the desk in front of you.\nYou will now be asked to recall all "+string(num_images_per_scene)+" items associated with each scene.\n\nYou can use more than one word to describe the object if necessary.\n<b>Use one line per object and the 'Enter' key to move to the next line.</b>\n\nYou will have up to "+string(time_limit)+" seconds to answer for each scene.\n<b>Use this time to try and recall <u>all</u> the items.</b>\n\nHit <b>Esc</b> to continue to the next scene.\n<b>Let the experimenter know now if you don't know where this key is.</b>\nTo begin, press the Esc button.");Instruct_text.redraw();
		NullTrial_wPort.get_stimulus_event(1).set_event_code("Thanks for trying!\n\nBefore we're done, we have just one more tests left.\n\nPlease grab the keyboard from the desk in front of you.\nYou will now be asked to recall all "+string(num_images_per_scene)+" items associated with each scene.\n\nYou can use more than one word to describe the object if necessary.\n<b>Use one line per object and the 'Enter' key to move to the next line.</b>\n\nYou will have up to "+string(time_limit)+" seconds to answer for each scene.\n<b>Use this time to try and recall <u>all</u> the items.</b>\n\nHit <b>Esc</b> to continue to the next scene.\n<b>Let the experimenter know now if you don't know where this key is.</b>\nTo begin, press the Esc button.");Instruct_text.redraw();
		array<double> tempport[1];NullTrial_wPort.get_stimulus_event(1).get_port_codes(tempport);
		NullTrial_wPort.get_stimulus_event(1).set_port_code(27);
		screentype.set_port_code(13);
		InstructPicWithoutImage.present();NullTrial_wPort.present();
		screentype.set_port_code(0);
		NullTrial_wPort.get_stimulus_event(1).set_port_code(int(tempport[1]));
		Instruct_text.set_caption(temptext);Instruct_text.redraw();
		system_keyboard.set_log_keypresses( true );
		system_keyboard.set_delimiter( 8 );
		system_keyboard.set_max_length( 1 );
		NullTrial_wPort.get_stimulus_event(1).set_port_code(33);
		bool skip;
		loop until false begin
			string press = system_keyboard.get_input();
			term.print_line(press);
			if press == "" then
				break;
			elseif press == "~" then
				NullTrial_wPort.get_stimulus_event(1).set_port_code(96);
				skip=true;
				break;
			end;
		end;
		NullTrial_wPort.present();
		if !skip then		
			default.present();
			array<int> rand_ord[num_scenes];rand_ord.fill(1,num_scenes,1,1);
			rand_ord.shuffle();
			string filename=tempsoundforfilename.filename().replace("stim\\1.wav","log\\S"+SubjectNum+"-"+RunType+"_Debrief.txt");
			loop int i=1 until !file_exists(filename) begin		if i==1 then 			filename=filename.replace(".txt","_"+string(i)+".txt");		else			filename=filename.replace("_"+string(i-1)+".txt","_"+string(i)+".txt");		end;		i=i+1;	end;
			out_results = new output_file;	
			out_results.open(filename);
			term.print_line(filename);
			out_results.print("Scene\tRecalledItems\n");
			term.print_line("Trial#\tScene\tRecalledItems");
			term.print_line("------\t-----\t-------------");
			
			loop int i=1 until i>rand_ord.count() begin
				NullTrial_wPort.get_stimulus_event(1).set_port_code(rand_ord[i]);
				NullTrial_wPort.present();
				string input_text=scene_item_recall(rand_ord[i],time_limit);
				NullTrial_wPort.present();
				term.print(string(i)+"/"+string(rand_ord.count())+"\t"+string(rand_ord[i])+"\t");
				out_results.print("Scene#"+string(rand_ord[i])+"\n");
				term.print_line(input_text.replace("\n","\t"));
				out_results.print(input_text);
				out_results.print("\n\n");
				i=i+1;
				default.present();wait_interval(500);
			end;
			out_results.close();		
		end;
			
		term.print_line("Total bonus for both parts was "+string(sum_bonus)+"$");
		loop until !instructions("Great!\n\nBefore we're done, we just want to ask a few questions.\n\n\nPlease wait for the experimenter to come in the room.",14) begin end;

		array<int>rand_order[sounds.count()];
		rand_order.fill(1,rand_order.count(),1,1);
		rand_order.shuffle();
		out_results = new output_file;	
		Just_for_the_dir_name=tempsoundforfilename.filename();
		Just_for_the_dir_name=Just_for_the_dir_name.replace("stim\\1.wav","log\\S"+SubjectNum+"-"+RunType+".txt");
		string filename=Just_for_the_dir_name.replace(".txt","_Debrief.txt");
		loop int i=1 until !file_exists(filename) begin		if i==1 then 			filename=filename.replace(".txt","_"+string(i)+".txt");		else			filename=filename.replace("_"+string(i-1)+".txt","_"+string(i)+".txt");		end;		i=i+1;	end;
		out_results.open(filename);
		term.print_line("Saving in file: "+filename+"; ");
		out_results.print("Sound\tHeardOrNot\tFilename\n");
		term.print_line("Sound\tHeard1OrNot2\tFilename");
		term.print_line("-----\t------------\t--------");

		if instructions("You will now hear a series of sounds and will be asked\nwhether you remember hearing them during sleep or not.\n\n\n\n(not all subjects are presented with sounds during sleep,\nso please pick the answer that reflects your personal experience).\n\n\n\Right-click to hear the first sound.",15) then
			loop int i=1 until i>rand_order.count() begin

				NullTrial_wPort.get_stimulus_event(1).set_port_code((rand_order[i]-1)/num_images_per_scene+1);
				NullTrial_wPort.get_stimulus_event(1).set_event_code("Debrief: Scene#"+string((rand_order[i]-1)/num_images_per_scene+1)+", object #"+string(mod(rand_order[i]-1,4)+1));
				sounds[rand_order[i]].present();NullTrial_wPort.present();
				wait_interval(25);
				NullTrial_wPort.get_stimulus_event(1).set_port_code(mod(rand_order[i]-1,4)+1);
				NullTrial_wPort.present();
				
				int res=was_sound_played_in_sleep_or_not();
				NullTrial_wPort.get_stimulus_event(1).set_port_code(res);NullTrial_wPort.present();
				wait_interval(50);
				term.print_line(string(rand_order[i])+"\t"+string(res)+"\t\t"+sounds[rand_order[i]].get_wavefile().filename());
				out_results.print(string(rand_order[i])+"\t"+string(res)+"\t"+sounds[rand_order[i]].get_wavefile().filename()+"\n");
				i=i+1;
				default.present();wait_interval(500);
			end;
		end;
		out_results.close();		
		term.print_line("Total bonus for both parts was "+string(sum_bonus)+"$");
		if instructions("Thank you!\n\n\nWe're done for today!\n\n\nRight-click to end.",12) then end;
		
	end;

	if RunType=="Sleep" then
		response_manager.set_button_active( 1, false);
		response_manager.set_button_active( 2, false);
		trial TMR = new trial ();
		TMR.set_type(first_response);
		text TMRtextcaption = new text();
		TMRtextcaption.set_caption("Press space to start TMR");
		TMRtextcaption.set_font_size(48);
		picture TMRtext=new picture();
		TMRtext.add_part(TMRtextcaption,0,0);
		
		# Initialization
		output_port port = output_port_manager.get_port( 1 );
		port.set_pulse_width( 20 );

		# TMR parameters    
		array<int> TMR_interval[3]={6000,6500,7000};#ms, between repeated stimuli
		int max_sleep_duration=120*60*1000; #ms, 120 minutes;
		double atten=0.1; #default cue attenuation
		
##############################CHANGE AFTER SOUNDS ARE SET###########################

		double min_atten_per_wav_file=1;
		double max_atten_per_wav_file=0;
		loop int i=1 until i>atten_per_wav_file.count() begin if atten_per_wav_file[i]>max_atten_per_wav_file then max_atten_per_wav_file=atten_per_wav_file[i]; end; if atten_per_wav_file[i]<min_atten_per_wav_file && atten_per_wav_file[i]>0 then min_atten_per_wav_file=atten_per_wav_file[i]; end; i=i+1;end;

	### SOUND LOADING/SAVING AND INITIALIZATION
	### ---------------------------------------

		string filename=TMRcues[1].get_wavefile().filename();
		filename=filename.replace("stim\\Stim0.wav","log\\TMR_list_for_"+SubjectNum+".txt");
		input_file in = new input_file;
		in.open(filename);
		in.set_delimiter( '\n' );
		array<string> inpt[0];
		#loop int i=1 until i>inpt.count()-1 begin inpt[i]=in.get_string(); i=i+1; end;
		loop int i=1 until !in.last_succeeded() begin inpt.resize(inpt.count()+1);inpt[i]=in.get_string(); i=i+1; end;
		inpt[inpt.count()]=string(num_scenes*num_images_per_scene+num_practice_images+1);
		TMRcues.resize(inpt.count());
		loop int i=1 until i>TMRcues.count() begin
			TMRcues[i]=sounds[item_scene_allocation[int(inpt[i])]];
			##!#TMRcues[i]=sounds[int(i)];
			#TMRcues[i].set_attenuation(atten);
			TMRcues[i].set_attenuation(atten+atten_per_wav_file[item_scene_allocation[int(inpt[i])]]);
			i=i+1;
		end;

		NullTrial_wPort.get_stimulus_event(1).set_event_code("atten="+string(atten));
		NullTrial_wPort.get_stimulus_event(1).set_port_code(0);
		NullTrial_wPort.present();



	### BUILD TRIAL SETS
	### ----------------
		
		#set up TMR
		array<int> rand_ord[inpt.count()];
		rand_ord.fill(1,rand_ord.count(),1,1);
		rand_ord.shuffle();
		int delay=0;
		loop int i=1 until i>=int(max_sleep_duration/TMR_interval[1])+1
		begin	
			stimulus_event tempstimev=TMR.add_stimulus_event(TMRcues[rand_ord[mod(i-1,rand_ord.count()+1)+1]]);	
			int rand_ITI=random(1,3);
			tempstimev.set_duration(TMR_interval[rand_ITI]+TMRcues[rand_ord[mod(i-1,rand_ord.count()+1)+1]].get_wavefile().duration());
			if i==1 then
				tempstimev.set_deltat(500);
			else
				tempstimev.set_deltat(TMR.get_stimulus_event(TMR.stimulus_event_count()-1).duration()+delay);
			end;
			delay=0;
			string string_i=string(i);
			loop until string_i.count()==4 begin string_i="0"+string_i; end;
			string txt2=inpt[rand_ord[mod(i-1,rand_ord.count()+1)+1]];
			##!#string txt2=string(rand_ord[mod(i-1,rand_ord.count()+1)+1]);
			if txt2.count()==1 then
				txt2="0"+txt2;
			end;
			tempstimev.set_event_code(string_i+"_TMR"+txt2+", initial_atten="+string(atten+atten_per_wav_file[item_scene_allocation[int(inpt[rand_ord[mod(i-1,rand_ord.count()+1)+1]])]])+", file="+TMRcues[rand_ord[mod(i-1,rand_ord.count()+1)+1]].get_wavefile().filename());
			tempstimev.set_port_code(int(inpt[rand_ord[mod(i-1,rand_ord.count()+1)+1]]));

			##!#tempstimev.set_port_code(rand_ord[mod(i-1,rand_ord.count()+1)+1]);
			tempstimev.set_target_button(4);
			i=i+1;
			if mod(i,rand_ord.count()+1)==0 then
				int last_stim=rand_ord[rand_ord.count()];
				rand_ord.shuffle();
				loop until rand_ord[1]!=last_stim begin rand_ord.shuffle(); end;
				delay=tempstimev.duration()-100;
				tempstimev=TMR.add_stimulus_event(new nothing);	
				tempstimev.set_deltat(100);
				string_i=string(i);
				loop until string_i.count()==4 begin string_i="0"+string_i; end;
				tempstimev.set_event_code(string_i+"_Finished round#"+string(i/(rand_ord.count()+1)));
				if i/(rand_ord.count()+1)<10 then
					tempstimev.set_port_code(84+i/(rand_ord.count()+1));
				else
					tempstimev.set_port_code(95);
				end;			
				tempstimev.set_target_button(4);
				i=i+1;
			end;
		end;
				
	### RUN THE EXPERIMENT
	### ------------------

		# Sleep + TMR
		if max_sleep_duration>NoiseNap.duration() then
			Noise_for_nap.set_loop_playback(true);
		end;
		Noise_for_nap.present();
		int timeout=clock.time()+max_sleep_duration;
		int total_num_cues_played=0;
		int total_non_cues=0;
		int total_activation_time=0;
		string laststim; int intlaststim; sound last_played;
		term.print_line("Nap start time = "+date_time("hh:nn:ss"));
		loop until clock.time()>timeout
		begin
			#TMRtextcaption.set_caption("Press space to start TMR\n\nF4 to quit\n\nCue intensity (volume) = "+string(double(int(-round(atten,2)*100))/100));
			TMRtextcaption.set_caption("Press space to start TMR\n\nF4 to quit\n\nCue intensity (volume) = "+string(round(-atten,2)));
			TMRtextcaption.redraw();
			TMRtext.present();
			TMRtextcaption.set_caption("Press space/enter to stop TMR");
			TMRtextcaption.redraw();
			int count3 = response_manager.total_response_count(3);
			int count4 = response_manager.total_response_count(4);
			int count6 = response_manager.total_response_count(6);
			int count7 = response_manager.total_response_count(7);
			loop until response_manager.total_response_count(3) > count3 || response_manager.total_response_count(6) > count6 || response_manager.total_response_count(4) > count4 || response_manager.total_response_count(7) > count7 || clock.time()>timeout begin end;		
			int was_there_a_keypress=0;
			int start_time;
			if response_manager.total_response_count(4) > count4 then
				was_there_a_keypress=1;
				TMRtext.present();
				TMR.set_duration(timeout-clock.time());
				start_time=clock.time();
				term.print_line("\nTMR start time = "+date_time("hh:nn:ss"));
				TMR.present();
				term.print_line("TMR stop time = "+date_time("hh:nn:ss"));
			elseif response_manager.total_response_count(6) > count6 || response_manager.total_response_count(7) > count7 then
				if response_manager.total_response_count(6) > count6 then
					if atten+0.01+max_atten_per_wav_file<1 then
						atten=atten+0.01;
						wait_interval(25);
						port.send_code(128);
					end;
				elseif atten-0.01+min_atten_per_wav_file>0 then
					atten=atten-0.01;
					wait_interval(25);
					port.send_code(128);
					wait_interval(25);
					port.send_code(128);
				end;
				loop int i=1 until i>TMRcues.count() begin					
					TMRcues[i].set_attenuation(atten+atten_per_wav_file[item_scene_allocation[int(inpt[i])]]);
					#TMRcues[i].set_attenuation(atten);
					i=i+1;
				end;
				if laststim.count()>0 then
					last_played.set_attenuation(atten+atten_per_wav_file[item_scene_allocation[intlaststim]]);
				end;
				#TMRtextcaption.set_caption("Press space to start TMR\n\nF4 to quit\n\nCue intensity (volume) = "+string(double(int(-round(atten,2)*100))/100));
				TMRtextcaption.set_caption("Press space to start TMR\n\nF4 to quit\n\nCue intensity (volume) = "+string(round(-atten,2)));
				TMRtextcaption.redraw();
				NullTrial_wPort.get_stimulus_event(1).set_event_code("atten="+string(atten));
				wait_interval(50);
				NullTrial_wPort.present();
				TMRtext.present();
			elseif response_manager.total_response_count(3) > count3 && laststim.count()>0 then
				TMRtext.present();
				term.print_line("\nTMR start time = "+date_time("hh:nn:ss"));
				wait_interval(500);
				NullTrial_wPort.get_stimulus_event(1).set_event_code("9999_TMR"+laststim+", initial_atten="+string(atten+atten_per_wav_file[item_scene_allocation[intlaststim]])+", file="+last_played.get_wavefile().filename());
				NullTrial_wPort.get_stimulus_event(1).set_port_code(intlaststim);
				last_played.present();NullTrial_wPort.present();
				wait_interval(last_played.get_wavefile().duration());
				term.print_line("TMR stop time = "+date_time("hh:nn:ss"));
				term.print("Presented previous stimulus again once [snd#(filename)]: "+laststim+"("+NullTrial_wPort.get_stimulus_event(1).event_code().substring(NullTrial_wPort.get_stimulus_event(1).event_code().count()-5,2)+")");
				NullTrial_wPort.get_stimulus_event(1).set_port_code(0);
			end;

			if was_there_a_keypress==1 then
				string tmp=stimulus_manager.last_stimulus_data().event_code();
				term.print("Presented stimuli [snd#(filename)]: ");
				tmp.resize(4);
				int num_cues_played=int(double(tmp))-total_num_cues_played-total_non_cues;
				if tmp=="# of" || tmp=="atte" then
						num_cues_played=0;
				end;
				loop int i=1 until i>num_cues_played begin
				##!#loop int i=1 until i>num_cues_played-1 begin
					if TMR.get_stimulus_event(1).event_code().substring(9,2)=="is" then
						num_cues_played=num_cues_played-1;
						total_non_cues=total_non_cues+1;
						TMR.remove_stimulus_event(1);
						term.print(", CYCLE_END");
					else
						if i!=1 then term.print(", "); end;
						term.print(TMR.get_stimulus_event(1).event_code().substring(9,2)+"("+TMR.get_stimulus_event(1).event_code().substring(TMR.get_stimulus_event(1).event_code().count()-5,2)+")");
						laststim=TMR.get_stimulus_event(1).event_code().substring(9,2);
						TMR.remove_stimulus_event(1);
						i=i+1;
					end;
				end;
				term.print_line(" ");
				total_num_cues_played=total_num_cues_played+num_cues_played;
				##!#total_num_cues_played=total_num_cues_played+num_cues_played-1;
				total_activation_time=total_activation_time+clock.time()-start_time-500;
				term.print_line("# of cues played in last activation: "+string(num_cues_played)+"\n# of cues played throughout nap so far: "+string(total_num_cues_played)+"\nLength of last activation: "+get_min_sec(clock.time()-start_time-500)+" minutes\nTotal activation throughout nap so far: "+get_min_sec(total_activation_time)+" minutes");
				NullTrial_wPort.get_stimulus_event(1).set_event_code("# of cues played in last activation: "+string(num_cues_played)+"; # of cues played throughout nap so far: "+string(total_num_cues_played)+"; Length of last activation: "+get_min_sec(clock.time()-start_time-500)+" minutes; Total activation throughout nap so far: "+get_min_sec(total_activation_time)+" minutes");
				wait_interval(100);
				NullTrial_wPort.present();
				stimulus_event temp_stim_event=TMR.get_stimulus_event(1);
				temp_stim_event.set_deltat(500);
				intlaststim=int(laststim);
				if laststim=="08" then #resolving the weirdest bug in the world, where int("08") and int("09") is zero.
					intlaststim=8;
				elseif laststim=="09" then
					intlaststim=9;
				end;
				last_played=sounds[item_scene_allocation[intlaststim]];
				last_played.set_attenuation(atten+atten_per_wav_file[item_scene_allocation[intlaststim]]);
			end;

		end;
		TMRtextcaption.set_caption(string(double(max_sleep_duration)/(60*1000)) + " minutes are over");
		TMRtextcaption.redraw();
		TMRtext.present();
		Silencetrial.present();
		NullSound.present();			
	end;


