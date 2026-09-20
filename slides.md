Title: GPGPU Project - Autumn 2025Slide 1

    Title: GPGPU Project
    Content: Autumn 2025 - GISTRE/SCIA S9

Slide 2

    Title: Objective
    Content: GStreamer plugin for foreground/moving object separation in videos.
        INPUT = video stream (stabilized) / OUTPUT = video with differences in red.
        Preliminary step in many processing chains.
        Many local operations => Good candidate for GPU optimization.

Slide 3

    Title: What you need to know about GStreamer
    Content:
        It's a pain to get working.
        It's very efficient.
        You only need to look at the content of the function void compute_cu(ImageView<rgb8> in).
        compute_cu is called for each new frame.
        in contains the data of the current frame.
        You must use in to update the internal state of your system → static variables to ensure persistence between frames.
        You must modify in to return the frame that will be displayed or saved ("zero copy" architecture).

Slide 4 & 5

    Title: General Approach
    Content: (Diagrammatic flow describing the overall process)
        Input: Frame t (img data + t)
        Process: Background estimation process (Internal BEP state, Previous values…) → Mask cleaning process (Clean change mask at t) → Alerting process (Alert indicator yes/no)
        Output: Background image at t, Change mask at t, Difference.

Slide 6

    Title: Detail of the steps
    Content: (Blank slide, title suggests subsequent slides detail the steps)

Slide 7

    Title: Background estimation process
    Content: Pixel-wise iterative process, for each frame (For each pixel position p).
        Based on “weighted reservoir sampling”, an online sampling algorithm.
        States: rs = K reservoir values and weights, randState = 1 rand state.
        Logic: Find matching reservoir, then update weights and samples (matching, empty slot, or weighted reservoir replacement). Cap weights and set the background to the rgb with max weight.

Slide 8

    Title: m_idx = find_matching_reservoir(p, rs)
    Content: Finds the first matching reservoir (color difference < RGB_DIFF_THRESHOLD) or an empty reservoir, or returns -1.
        💡 Warning: Be careful when computing the difference with unsigned integers: cast them to signed integer first!

Slide 9

    Title: Handling random number generators
    Content:
        Maintain a random state for each pixel position: static Image<curandState> rng_states;
        Initialization: using a kernel which calls curand_init(seed, global_pixel_pos, 0, &randState_row[x]);
        Update: using float rand_val = curand_uniform(&randState_row[x]); (0 ≤ rand_val ≤ 1).

Slide 10

    Title: 2. Mask cleaning process overview
    Content:
        Calculation of the motion mask (Result of the previous process, a map of motion scores ≥ 0 per pixel).
        Steps: Noise suppression (Morphological opening by a radius 3 disk) → Hysteresis thresholding (Low threshold: 4, High threshold: 30) → Masking (input + 0.5 * red * mask).

Slide 11

    Title: 2.1 Noise suppression
    Content: Morphological opening (erosion followed by dilation).
        Erosion: new value for p(x,y) = min value in neighborhood of p(x,y)
        Dilation: new value for p(x,y) = max value in neighborhood of p(x,y)

Slide 12

    Title: 2.2. Hysteresis thresholding
    Content:
        Principle: Suppress weak signals. Propagate strong signals towards medium signals.

Slide 13

    Title: 2.2. Implementation of hysteresis reconstruction
    Content:
        Idea: Marker pixels are propagated into the mask until stability. Markers are initialized with elements > high threshold. Input contains all elements > low threshold.
        (Includes a C++/CUDA code snippet for the reconstruction kernel and main loop.)

Slide 14

    Title: Provided material
    Content: (Blank slide)

Slide 15

    Title: Gstreamer code
    Content: Base code provided on Moodle. Implement a GStreamer CUDA and CPP filter, ideally integrating these parameters:
        bg=uri : uri to a background image (default="" => estimated)
        opening_size=(int) : size of the opening
        th_low=(int) : low filter value (default=3)
        th_high=(int) : high filter value (default=30)
        bg_sampling_rate=(int) : frame sampling interval for background estimation (default=500ms)
        bg_number_frame=(int) : number of frames used for background estimation (default=10)

Slide 16

    Content: Demo
        (Includes a GStreamer command line for demonstration)

Slide 17

    Title: Requirements (Attendus)
    Content: (Blank slide)

Slide 18

    Title: Evaluation Criteria / Advice
    Content:
        Correct code => ACCEPTABLE qualitative results. (Results will not be optimal with this method).
        Speed: the faster the framerate, the better.
        Advice: Have a functional C++ version (baseline). Use Git tags for versions. Perform optimizations one by one.

Slide 19

    Title: Deliverables (Livrables)
    Content:
        Implementation (Source code for C++ CPU reference, CUDA implementation(s), benchmark tools, Build scripts). Results must be reproducible.
        Brief Report (Description of the subject, task distribution, benchmarks/graphs, performance/bottleneck analysis).
        Defense Slides
        Group distribution (on Moodle) => today at the end of the lab session.

Slide 20

    Title: Defenses
    Content:
        Schedule: December 18th and 19th.
        Format: 15’ presentation, 5’ demo (Data: 

), 5’ discussion.
Logistics: Defenses on Teams (links sent the week before).
Group Size: Project by group of 4. All members must be present.
Submission Deadline: You must submit all files on Dec 17th in the evening.
