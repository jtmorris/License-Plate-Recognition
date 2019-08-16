import numpy as np
import cv2
import pytesseract
import imutils
from libs.four_point_transform import four_point_transform
from skimage.filters import threshold_local
from time import sleep
import argparse
import libs.jtm_python_helpers as JPH


class License_Plate:
	tunings = {
		# Ratio of plate candidate width to its height.
		## United States plates are twice as wide as tall. A ratio
		## closer to 2 will be more discerning, but will become
		## subject to rejecting candidate based on minor skewing of
		## capture angle or of contour selection and approximation
		## error. (min, max): First element is minimum ratio. Second is
		## maximum ratio.
		"w2h_ratio_range": (1.5, 2.5),

		# Perimeter coefficient for use in approximating polygon.
		##### Larger coefficient means more lenient approximation.
		"poly_epsilon_coeff": 0.018,

		# A range of acceptable character counts when assessing contours.
		##### (min, max). If max is None, no upper bound will be used.
		##### A higher minimum, approaching minimum number of plate
		##### characters will be more discerning, but we haven't done
		##### pre-processing for accurate OCR yet, so OCR may be very
		##### inaccurate.
		"rough_ocr_char_range": (2, None),

		# A list of known municipality names to appear on license plates.
		# Used to filter out OCR return lines that are junk.
		"known_municipalities": ("Alabama", "Alaska", "American Samoa",
			"Arizona", "Arkansas", "California", "Colorado", "Conneticut",
			"Delaware", "District of Columbia", "Florida", "Georgia",
			"Guam", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
			"Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
			"Massachusetts", "Michigan", "Minnesota", "Mississippi",
			"Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
			"New Jersey", "New Mexico", "New York", "North Carolina",
			"North Dakota", "C.N.M.I", "Ohio", "Oklahoma", "Oregon",
			"Pennsylvania", "Puerto Rico", "Rhode Island",
			"South Carolina", "South Dakota", "Tennessee", "Texas",
			"Utah", "Vermont", "Virgin Islands", "Virginia",
			"Washington", "West Virginia", "Wisconsin", "Wyoming",

			# Butchered potential alternatives based on styling
			"ichigan", # Some Michigan plates have a strange "M"
			),

		# A list of known slogans, mottos, and other markings. Used to
		# filter out OCR return lines that are junk.
		## Program will look for occurrences of these slogans in lines
		## of the extracted OCR. The slogan specified below can be
		## parts of a larger slogan. Recommend using minimum required
		## specificity. For example, "100 Years Centennial" could be
		## just "Centennial" as Centennial is an invalid license plate
		## number, and many states have variations of centennial plates.
		## Comparison will be case insensitive.
		"known_slogans": (
			"Disabled", "School Vehicle", "Not for Hire", "State Patrol",
			"Commercial", "Motorcycle", "Municipal", "Preserve", "Dealer",
			"Centennial", "Sesquicentennial", "In God We Trust",
			"Trailer", "Tractor", "Bicentennial", "Live Free or Die",
			"We Remember", "Anniversary",

			# Alaska
			"Last Frontier", "North to the Future",
			# American Samoa
			"Motu O Fiafiaga",
			# Arizona
			"Grand Canyon State",
			# Arkansas
			"Land of Opportunity", "The Natural State",
			# California
			"dmv.ca.gov", "Yosemite National Park",
			"We Will Never Forget",
			# Colorado

			# Conneticut
			"Constitution State",
			# Delaware
			"The First State"
			# D.C.
			"Nation's Capital", "A Capital City", "Celebrate",
			"Discover", "Taxation Without Representation", "dc.gov",
			"Inauguration",
			# Florida
			"MyFlorida.com", "Sunshine State",
			# Georgia
			"Peach State",
			# Guam
			"Hafa Adai", "America's Day Begins", "Hub of the Pacific",
			"Tano Y Chamorro",
			# Hawaii
			"Aloha State",
			# Idaho
			"Famous Potatoes",
			# Illinois
			"Land of Lincoln",
			# Iowa

			# Kansas
			"The Wheat State", "Midway USA",
			# Kentucky
			"Bluegrass State", "that friendly", "Unbridled Spirit",
			# Louisiana
			"Sportsman's Paradise",
			# Maine
			"Vacationland",
			# Maryland
			"War of 1812", "www.maryland.gov", "the Chesapeake",
			# Massachusetts
			"The Spirit of America",
			# Michigan
			"Water Wonderland", "Water-Winter", "Great Lake State",
			"Great Lakes", "michigan.org", "Motor Capital",
			"Great Lakes", "Spectacular Peninsulas",
			"The Mackinac Bridge",
			# Minnesota
			"10,000 Lakes", "10000 Lakes",
			# Mississippi

			# Missouri
			"Show-Me State", "Show Me State",
			# Montana
			"Treasure State", "Big Sky", "100 Years",
			# Nebraska
			"Beef State", "Cornhusker", "state.ne.us",
			"nebraska.gov",
			# Nevada
			"Silver State", "Home Means Nevada",
			# New Hampshire
			"LiveFreeOrDie", "Garden State",
			# New Mexico
			"Land of Enchantment", "New Mexico USA", "Sunshine State",
			"Chile Capital of the World",
			# New York
			"Empire State",
			# North Carolina
			"First in Freedom", "First in Flight",
			# North Dakota
			"Peace Garden State", "Discover the Spirit", "Legendary",
			# North Mariana Islands
			"Hafa Adai",
			# Ohio
			"Seat Belts Fastened", "heart of it all",
			"Birthplace of Aviation", "DiscoverOhio.com"
			# Oklahoma
			"Oklahoma is OK", "OK!", "Native America",
			"Explore Oklahoma", "TravelOK.com",
			# Oregon
			"Pacific Wonderland",
			# Pennsylvania
			"Keystone State", "Got a Friend", "state.pa.us",
			"visitPA.com",
			# Puerto Rico
			"Isla del Encanto", "Cincuentenario",
			"Puerto Rico does it better", "Puerto Rico lo hace mejor",
			# Rhode Island
			"Ocean State", "300th Year"
			# South Carolina
			"Smiling Faces", "Beautiful Faces", "Travel2SC.com",
			"While I Breathe",
			# South Dakota
			"Great Faces", "Great Places",
			# Tennessee
			"Volunteer State", "Sounds Good to Me", "tnvacataion.com",
			# Texas
			"Lone Star", "Years of Statehood",
			# Utah
			"Greatest Snow", "Life Elevated", "United we Stand",
			# Vermont
			"Green Mountain", "Green Mountains",
			# US Virgin Islands
			"Vacation", "Adventure", "American", "Paradise",
			"Our Islands", "Our Home", "Caribbean",
			# Virginia
			"Jamestown", "Is For Lovers",
			# Washington
			"Evergreen",
			# West Virginia
			"Mountain State", "Wonderful", "callwva.com",
			# Wisconsin
			"Dairyland",
			# Wyoming

		),
	}

	def __init__(self, image_of_plate):
		self.raw_image = image_of_plate
		self.proc_steps = []
		self.rejected_candidates = []
		self._store_proc_step(self.raw_image.copy(), "Raw image.")

	def run_plates(self):
		contour_candidates = self.find_contour_candidates()
		candidate_imgs = self.preprocess_candidates(contour_candidates)
		raw_strs = [self.ocr_image(img) for img in candidate_imgs]
		pp_strs = [self.ocr_postprocess(s) for s in raw_strs]
		vd_strs = [self.validate_string_us(s) for s in pp_strs]

		return vd_strs


	@classmethod
	def ocr_image(cls, image_to_ocr):
		"""Runs optical character recognition on an image.

		Uses PyTesseract to optically recognize characters in the
		provided image and returns result as a string.

		Args:
			image_to_ocr:	An OpenCV image. Should contain only text
							you want parsed.

		Returns:
			string: A string with detected characters.
		"""

		return pytesseract.image_to_string(image_to_ocr,
										   config='--psm 11')


	@classmethod
	def validate_string_us(cls, the_string: str):
		"""Makes sure string could be a United States License Plate.

		Analyzes passed string to make sure it isn't obviously not a
		license plate.

		Args:
			the_string (str): The string to check.
		Returns:
			tuple:	First element is a bool. True if string seems to
					conform to license plate norms. False if obviously
					incorrect. Second element is a string with the
					reason for rejection if check failed. String is
					empty if check succeeded.

		Examples:
			>>> License_Plate.validate_string_us("SAMPLE1")
			(True, '')
			>>> License_Plate.validate_string_us("FOO")
			(False, 'Too short.')
			>>> License_Plate.validate_string_us("Bar")
			(False, 'Too short.')
			>>> License_Plate.validate_string_us("expialidocious")
			(False, 'Too long.')

		"""

		# Check #1:  Length
		# 	United States license plates typically have 5 to 7
		# 	characters. Some plates (vanity and specialty) have between
		# 	2 and 8 characters. To make this useful, we're going to set
		# 	a lower limit of 4 and an upper limit of 8.
		s_len = len(the_string)
		if s_len > 8:
			return (False, "Too long.")

		if s_len < 4:
			return (False, "Too short.")

		return (True, "")



	def find_contour_candidates(self, contour_proc_limit: int = 10):
		"""Returns contours that are possibly license plates.

		Uses basic image analysis to identify potential license
		plate candidates, gets their contour, and returns a list.
		Note that these are NOT necessarily license plates. Further
		refined	analysis should be done.

		Args:
			contour_proc_limit (int):	The max number of contours
				to do the costly operations on.

		Returns:
			list:	A list of OpenCV contours likely to be license
					plates.

		Theory:
			We're going to use known characteristics of license
			plates to filter out obviously incorrect data. First
			characteristic is size. Second characteristic is shape.
			Third characteristic is letters.

			Size: License plates are going to be among the biggest
			contours picked out, as they are large and uniform in
			shape. We will sort contours by size, pick a handful off
			the top, and continue analysis with them.

			Shape: License plates are rectangular, or, if slightly
			skewed, a 4-sided polygon. Assuming the plate is
			horizontal, the width of the plate is larger than the
			height by a sizable margin (in United States, 12 inches
			wide by 6 inches high). We will check contour boundary
			dimensions, and the plate may not be perfectly
			horizontal, so we will use a threshold near the 2:1
			ratio, but with a comfortable enough margin.

			Letters: For accurate plate reading, we will want to do
			a bunch of further processing first. We simply want to
			see if OCR finds anything that could be letters. To do
			so, we will OCR the contour, strip out whitespace, and
			see if we have at least two characters.
		"""
		OP = JPH.Performance_Monitor()
		img = self.raw_image
		# Step 1: Edge detection.
		T = JPH.Performance_Monitor()
		edged_img = cv2.Canny(img, 30, 200)
		T.stop_and_spit(context=
						"[find_contour_candidates(): Edge detection.]")
		exec_time = T.get_time("nanoseconds")
		self._store_proc_step(edged_img, "Edge detection", exec_time)

		T.start()
		# Step 2: Extract all contours
		all_contours = cv2.findContours(edged_img.copy(), cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
		all_contours = imutils.grab_contours(all_contours)
		print("[INFO] [find_contour_candidates()] " +
			str(len(all_contours)) + " found initially.")
		T.stop_and_spit()

		T.start()
		# Step 3: Sort contours by size, and limit the number to analyze
		size_filtered_contours = sorted(all_contours, key=cv2.contourArea,
										reverse=True)[:contour_proc_limit]
		num_eliminated = len(all_contours) - len(size_filtered_contours)
		print("[INFO] [find_contour_candidates()] " + str(num_eliminated) +
			" eliminated based on sorting and limiting.")
		T.stop_and_spit(context=
						"[find_contour_candidates(): Sorting and limiting.] ")

		# Step 4: Filter by shape. This is done in a helper function
		#         to facilitate mapping, rather than looping.
		# If helper function declares it passes (returns True), save
		# the contour. Otherwise ignore it.
		T.start()
		shape_filtered_contours = ([c for c in size_filtered_contours
									if self._contour_shape_passes(c)])
		num_eliminated = (len(size_filtered_contours) -
			len(shape_filtered_contours))
		print("[INFO] [find_contour_candidates()] " + str(num_eliminated) +
			" eliminated based on shape filtering.")
		T.stop_and_spit(context=
						"[find_contour_candidates(): Shape filtering.] ")

		T.start()
		# Step 6: Check for letters. Final check!
		ocr_filtered_contours = ([c for c in shape_filtered_contours
									if self._contour_ocr_passes(img, c)])
		num_eliminated = (len(shape_filtered_contours) -
			len(ocr_filtered_contours))
		print("[INFO] [find_contour_candidates()] " + str(num_eliminated) +
			" eliminated based on OCR filtering.")
		T.stop_and_spit(context=
						"[find_contour_candidates(): OCR filtering.] ")

		OP.stop_and_spit(context=
						"[find_contour_candidates()] Overall run time: ")
		op_t = OP.get_time("nanoseconds")

		# Draw contours on the image and save
		c_img = self.raw_image.copy()
		(cv2.drawContour(c_img, c, -1, (255, 255, 0), 1)
			for c in ocr_filtered_contours)
		self._store_proc_step(c_img, "Find contour candidates.", op_t)

		# Done! Yay! Return the contours we got.
		return ocr_filtered_contours

	@classmethod
	def _contour_shape_passes(cls, contour):
		"""Checks if contour shape matches filter criteria.

		A helper function for find_contour_candidates() which
		assesses the contour shape and makes sure it is a
		4-sided polygon with a contour width sufficiently larger
		than its height to satsify tuning parameters.

		Args:
			contour: An OpenCV contour to check the shape of.

		Returns:
			bool: True if shape passes. False if not.
		"""
		ec = cls.tunings["poly_epsilon_coeff"]
		(wr_min, wr_max) = cls.tunings["w2h_ratio_range"]

		perimeter = cv2.arcLength(contour, True)
		approx_polygon = cv2.approxPolyDP(contour, ec * perimeter, True)
		if len(approx_polygon) == 4: # We have a parallelogram
			# Make sure width is more than height in accoradance with
			# tuning ratio
			(w,h) = cv2.boundingRect(approx_polygon)[-2]
			wr = w*1.0 / h
			if wr_min <= wr <= wr_max: # We have a candidate!
				return True

		return False

	@classmethod
	def _contour_ocr_passes(cls, image, contour):
		"""Checks if a contour has characters within acceptable criteria.

		A helper function for find_contour_candidates() which
		assesses the interior of a contour and sees if it contains
		characters within the specified tuning parameters.
		"""
		(min_chrs, max_chrs) = cls.tunings["rough_ocr_char_range"]

		roi = cls.extract_roi(image, contour)

		text = cls.ocr_image(roi)
		# Remove all whitespace
		text = ''.join(text.split())

		# In range?
		if max_chrs:
			return (min_chrs <= text <= max_chrs)

		return (min_chrs <= text)

	def preprocess_candidates(self, contour_candidates):
		"""Returns list of images preprocessed for effective OCR.

		Extracts image of the license plate specifically, runs several
		preprocessing steps in preparation for successful OCR, and returns
		the images as a list.

		Args:
			contour_candidates: A list of OpenCV contours to extract and
				preprocess.
		Returns:
			list: A list of OpenCV images of license plates preprocessed
				for OCR.
		"""
		T = JPH.Performance_Monitor()
		img = self.raw_image

		count = 0

		pped = list()
		for contour in contour_candidates:
			count = count + 1
			roi_str = "ROI #" + str(count)
			# Step 1: Extract image
			T = JPH.Performance_Monitor()
			lp = self.extract_roi(img, contour).copy()
			T.stop_and_spit(context=
				"[preprocess_candidates(): Extract image]")
			self._store_proc_step(lp, roi_str, T.get_time("nanoseconds"))


			# Step 2: Transform image to normal to camera axis
			T.start()

			lp_gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
			lp_gray = cv2.GaussianBlur(lp_gray, (11,11), 0)
			lp_edge = cv2.Canny(lp_gray, 75, 200)
			lp_cnts = cv2.findContours(lp_edge.copy(), cv2.RETR_LIST,
										cv2.CHAIN_APPROX_SIMPLE)
			lp_cnts = imutils.grab_contours(lp_cnts)
			lp_cnts = sorted(lp_cnts, key=cv2.contourArea, reverse=True)[:5]
			lp_cnts = [c for c in lp_cnts if self._contour_is_rectangle(c)]
			lp_cnt = lp_cnts[0] # Assume largest rectangle is plate
			warped = four_point_transform(lp, lp_cnt.reshape(4, 2))

			T.stop_and_spit(context=
				"[preprocess_candidates(): 4-point transform]")
			self._store_proc_step(warped,
				roi_str + ": 4-point transform", T.get_time("nanoseconds"))

			# Step 3: Binarize the image
			T.start()
			offset 		=	self.tunings["threshold_offset"]
			block_size	=	self.tunings["threshold_block_size"]
			binarized = threshold_local(warped, block_size, offset=offset)
			T.stop_and_spit(context=
				"[preprocess_candidates(): Binarization]")
			self._store_proc_step(binarized.copy(),
				roi_str + ": binarized", T.get_time("nanoseconds"))
			pped.append(binarized)

		return pped

	@classmethod
	def ocr_postprocess(cls, ocr_result: str):
		"""Clean up license plate OCR results.

		Runs some postprocessing on an OCR result to remove unwanted
		text that was probably picked up, such as state names and
		slogans.

		Args:
			ocr_result (str): A string to run OCR postprocessing on.
		Returns:
			str: The string after postprocessing.

		Theory:
			United States license plates have a lot of extra
			junk on them that we don't care about, such as state
			names, registration stickers, mottos, et cetera. We
			want to get rid of all that junk. We're going to use
			the fact that what we want will be on its own line,
			as well as potential known possibilities of what is
			on the junk lines.

			Lines: If we only have one line, we're done. If we
			have 3 lines, it's probably the middle one. However,
			we should do the other checks, just in case. But,
			remember if we have 3 lines and that the middle is
			probably it.

			Municipalities: If a line contains full municipality
			(e.g. state) names, it's garbage. These known
			municipalites are defined in tunings.

			Slogans: If a line contains known mottos, tags, or
			slogans, it's garbage. For example, Alaska plates
			say "THE LAST FRONTIER" and New Mexico plates say
			"LAND OF ENCHANTMENT." These known mottos are
			defined in tunings.

			Whitespace: Whitespace on the ends is not useful.
		"""
		print("[INFO] [ocr_postprocess()] String before postprocessing:\n" +
			ocr_result + "'")
		OP = JPH.Performance_Monitor()
		lines = ocr_result.splitlines()
		og_lines = len(lines)
		if lines < 1: # WTF? Just phone home.
			print("[INFO] [ocr_postprocess()] Empty OCR text?")
			return ocr_result

		# Step 1: Lines
		if og_lines == 1: # This has got to be it
			return lines[0]
		elif og_lines == 3: # It's probably the middle one. Save it.
			fallback = lines[1]


		# Step 2: Eliminate municipality and slogan lines
		T = JPH.Performance_Monitor()
		muns  = cls.tunings["known_municipalities"]
		slogs = cls.tunings["known_slogans"]
		lines = [ln for ln in lines if any((muns in ln) or (slogs in ln))]
		print("[INFO] [ocr_postprocess()] Eliminated " +
			str(og_lines - len(lines)) +
			" lines based on municipality and/or slogan.")
		T.stop_and_spit(context=
			"[ocr_postprocess(): Eliminate municipalities and slogans] ")

		# Step 3: If we're good, good. If not and we have a fallback,
		#	fallback. Otherwise, we have nothing to show.
		if len(lines) == 1:
			print("[INFO] [ocr_postprocess()] " +
				"One line after postprocessing. " +
				"OCR postprocess best case scenario.")
			retval = lines[0].strip()
		elif fallback:
			retval = fallback.strip()
		else:
			retval = ocr_result.strip()

		print("[INFO] [ocr_postprocess()] " +
			"Resulting OCR string after postprocessing:\n'" +
			retval + "'")
		OP.stop_and_spit(context="[ocr_postprocess()] ")
		return retval

	@classmethod
	def _contour_is_rectangle(cls, contour, epsilon: float = None):
		"""Checks if contour provided is a four-sided polygon.

		Approximates the specified contour to a polygon, using
		cv2.approxPolyDP() and checks if the approximated polygon is
		four-sided.

		Args:
			contour: An OpenCV contour to check.
			epsilon: Default None. If None, one is calculated based on
				tuning parameters. If provided, the epsilon value is
				passed to cv2.approxPolyDP().

		Returns:
			bool: True if four-sided polygon. False if not.
		"""

		OP = JPH.Performance_Monitor()
		if not epsilon:
			perimeter = cv2.arcLength(contour, True)
			epsilon = cls.tunings["poly_epsilon_coeff"] * perimeter

		polygon = cv2.approxPolyDP(contour, epsilon, True)

		OP.stop_and_spit()
		if len(polygon) == 4:
			return True

		return False


	@classmethod
	def extract_roi(cls, image, roi_contour):
		"""Extract only the region of interest (ROI) of an image.

		Extracts only the part of the OpenCV image contained within
		a bounding rectangle of the given OpenCV contour.

		Args:
			image:			An OpenCV image to extract from.

			roi_contour:	An OpenCV contour of the ROI.

		Returns:
			OpenCV Image:	The part of the image bounding the ROI.
		"""
		OP = JPH.Performance_Monitor()
		(x, y, w, h) = cv2.boundingRect(roi_contour)

		retval = image[y:y+h, x:x+w]
		OP.stop_and_spit()
		return retval


	#######################################################
	# Store intermediate steps for debugging and analysis #
	#######################################################
	def _step_store_abstraction(self, storage_location: list,
			img, msg: str, tme_in_ns: int = None,):
		"""An abstraction function for storing of steps in proper format

		Abstracts more specific storage functions by storing properly
		formatted image processing step information in the specified
		list. See specific storage functions (e.g. _store_proc_step())
		for more information.
		"""
		storage_location.append({"image": img,
			"description": msg, "exec_time": tme_in_ns})

	def _store_proc_step(self, the_image, process_description: str,
			execution_time_in_ns: int = None):
		"""Store the image and description for later retrieval.

		Stores an image, a description of its state, and how long it
		took for processing to that state into an array for later.
		Using this function, instead of manually adding data to the
		array ensures proper formatting and default entries for easy
		retrieval and processing later on.

		Args:
			the_image:					An image showing the results.
			process_description(str):	A very short description of what
				steps were taken to yield the image.
			execution_time_in_ns(int):	The time, in nanoseconds, the
				processing took to yield the image. Optional.
		"""
		OP = JPH.Performance_Monitor()
		self._step_store_abstraction(self.proc_steps, the_image,
			process_description, execution_time_in_ns)
		OP.stop_and_spit()

	#### Processing Steps
	def get_proc_step(self, index):
		# Check process step exists
		if not index in self.proc_steps:
			raise IndexError("Index " + str(index) + " not found.")
		s = self.proc_steps[index]

		if self._properly_formed_proc_step(s, True):
			return s

		print("[ERROR] [get_proc_step()] Invalidly formed process step:")
		JPH.var_dump(s)
		return None

	def get_last_proc_step(self):
		"""Returns the last entry off the process step stack."""
		return self.get_proc_step(-1)
	def get_last_proc_step_image(self):
		"""Returns image in the last entry on the process step stack."""
		return self.get_last_proc_step()["image"]

	@classmethod
	def _properly_formed_proc_step(cls, process_step,
			die_if_not: bool = False):
		"""Verifies the format of a processing step.

		Verifies the format of a processing step for storage or
		retrieval.
		"""
		# Check that process step details are properly formed
		s = process_step
		req_keys = ["image", "description", "exec_time"]
		missing_key = map((lambda k: k if not k in s else False), req_keys)
		if missing_key:
			if die_if_not:
				raise KeyError("Malformed process step. Missing expected\
					key '" + str(missing_key) + "'.")
			else:
				return False

		return True

	##### Rejected Candidates
	def _store_rejected_candidate(self, the_image, rejection_reason: str,
			time_wasted_in_ns: int = None):
		"""Store a rejected image and description for later retrieval.

		Stores an image, a description of its state, and how long it
		took for processing to that state into an array for later.
		Using this function, instead of manually adding data to the
		array ensures proper formatting and default entries for easy
		retrieval and processing later on. Rejected images may need
		to be stored to eliminate false positive filtering.

		Args:
			the_image:					An image showing the results.
			process_description(str):	A very short description of what
				steps were taken to yield the image.
			execution_time_in_ns(int):	The time, in nanoseconds, the
				processing took to yield the image. Optional.
		"""
		OP = JPH.Performance_Monitor()
		self._step_store_abstraction(self.rejected_candidates,
			the_image, rejection_reason, time_wasted_in_ns)
		OP.stop_and_spit()







# Run doctests
if __name__ == "__main__":
    import doctest
    doctest.testmod()
