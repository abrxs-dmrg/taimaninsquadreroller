import cv2
import numpy as np
import pyautogui
import time
import os
import sys
import glob
import argparse
import random
from sklearn.cluster import DBSCAN
import keyboard

# -----------------------------
# COMMAND LINE ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description='Gacha reroller with Character Detection')
parser.add_argument('-min_5_star_cards', type=int, default=3, help='Minimum number of 5* cards required')
parser.add_argument('-match_threshold', type=float, default=0.80, help='Template matching confidence threshold')
parser.add_argument('-roll_delay', type=float, default=3.0, help='Delay after clicking the button')
args = parser.parse_args()

# Directory Setups
TARGET_CHARS_DIR = 'target_characters'
DEBUG_DIR = 'debug_logs'
for d in [TARGET_CHARS_DIR, DEBUG_DIR]: os.makedirs(d, exist_ok=True)

# Template Patterns
STAR_PATTERN = "star_template*.png"
BTN_PATTERN = "recruit_button*.png"

# -----------------------------
# SCALE & VISION FUNCTIONS 
# -----------------------------
def nms(points, scores, shape, overlap=0.3):
    # Prevents counting the same star multiple times by filtering out overlapping bounding boxes.
    if not points: return []
    boxes = np.array([[x, y, x + shape[1], y + shape[0]] for (x, y) in points])
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - (w * h))
        order = order[np.where(ovr <= overlap)[0] + 1]
    return [points[i] for i in keep]

def find_scaled(screen_gray, templates, threshold):
    # Searches for templates at different sizes to support various screen resolutions. This should work on different resolutions without you manually changing any parameter.
    best_val, best_pts, best_shape, best_name = 0, [], None, None
    for scale in np.linspace(0.6, 1.2, 7): # Soporte multi-resolución
        for name, temp in templates:
            w, h = int(temp.shape[1] * scale), int(temp.shape[0] * scale)
            if h > screen_gray.shape[0] or w > screen_gray.shape[1]: continue
            resized = cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= threshold)
            pts = list(zip(*locs[::-1]))
            if pts:
                max_v = np.max(res)
                if max_v > best_val:
                    best_val, best_shape, best_name = max_v, resized.shape, name
                    best_pts = nms(pts, [res[y, x] for x, y in pts], resized.shape)
    return best_pts, best_shape, best_name, best_val

def get_star_groups(points):
    # Groups detected stars into individual cards using DBSCAN. Each cluster represents a char card
    if not points: return {}
    # 120 is the pixel distance threshold to consider stars as part of the same card, tbh this can be more adjusted with more test to avoid a bad count of stars.
    clustering = DBSCAN(eps=120, min_samples=1).fit(points)
    groups = {}
    for i, label in enumerate(clustering.labels_):
        if label not in groups: groups[label] = []
        groups[label].append(points[i])
    return groups

# -----------------------------
# VALIDATION LOGIC
# -----------------------------
def check_victory(screen_gray, star_pts, star_shape, target_temps):
    """
    The reroll will be considered completed when the next conditions success: 
    1. Identifies 5-star cards based on the number of star clusters.
    2. Verifies if a specific target character occupies a 5-star slot.
    """
    star_groups = get_star_groups(star_pts)
    
    # 1. Identify columns (X) of cards that are ACTUAL 5-star cards
    # The groups must contain 5 or more detected stars
    five_star_columns = []
    for label, pts in star_groups.items():
        if len(pts) >= 5:
            avg_x = sum(p[0] for p in pts) / len(pts)
            five_star_columns.append(avg_x)
    
    total_fives = len(five_star_columns)
    
    # --- DEBUG ---
    if total_fives > 0:
        cols_str = ", ".join([f"Pos:{int(x)}" for x in five_star_columns])
        print(f"Analysis: {total_fives} 5* cards detected at -> {cols_str}")

    # CONDITION A: Are there enough total 5-star cards?
    # If your searching 3 but you get 2, it is a False.
    has_enough_fives = (total_fives >= args.min_5_star_cards)

    # CONDITION B: Is the target character one of those 5-star cards?
    target_is_high_tier = False
    target_name_found = "None"
    
    if target_temps:
        char_pts, char_shape, char_name, _ = find_scaled(screen_gray, target_temps, 0.80)
        for (cx, cy) in char_pts:
            for fx in five_star_columns:
                # # Margin of 150px to align character with star column
                if abs(cx - fx) < 150:
                    target_is_high_tier = True
                    target_name_found = char_name
                    break

    # --- OUTPUT LOGIC ---
    if not target_temps:
        # General Mode: Only total count matters (Non character added as target)
        if has_enough_fives:
            return True, f"SUCCESS: {total_fives} 5* cards found."
    else:
        # Specific Mode: Both conditions must be met independently
        if has_enough_fives and target_is_high_tier:
            return True, f"¡SUCCESS!: {target_name_found} is 5* and there are {total_fives} in total."
        
        if target_is_high_tier and not has_enough_fives:
            print(f"Almost: {target_name_found} is 5*, but only {total_fives}/{args.min_5_star_cards} found.")
    
    return False, f"Searching... (5* detected: {total_fives}/{args.min_5_star_cards})"

# -----------------------------
# ACTIONS, AVOID BOT DETECTION RANDOMIZER
# -----------------------------
def human_click(pos, shape):
    # randomized clicking to avoid automated behavior detection.
    tx = pos[0] + random.randint(10, shape[1]-10)
    ty = pos[1] + random.randint(10, shape[0]-10)
    # Move to target
    pyautogui.moveTo(tx, ty, duration=random.uniform(0.4, 0.8), tween=pyautogui.easeOutQuad)
    pyautogui.click()

def load_temps(pattern):
    # Loads all PNG files matching the pattern as grayscale templates.
    return [(os.path.basename(f), cv2.imread(f, 0)) for f in glob.glob(pattern) if cv2.imread(f, 0) is not None]

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    star_temps = load_temps(STAR_PATTERN)
    btn_temps = load_temps(BTN_PATTERN)
    target_temps = load_temps(os.path.join(TARGET_CHARS_DIR, "*.png"))

    print(f"Starting... {'Target Character Mode' if target_temps else 'General REROLL Mode'}")
    attempt = 0

    while not keyboard.is_pressed('esc'):
        attempt += 1
        print(f"\n[Attempt {attempt}] Scanning...")
        
        # Capture screen
        img = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Look for stars
        s_pts, s_shape, _, _ = find_scaled(gray, star_temps, args.match_threshold)
        
        # Validate Victory Conditions
        success, msg = check_victory(gray, s_pts, s_shape, target_temps)
        print(f" -> {msg}")

        if success:
            print("¡SUCCESS!")
            break

        # If no success, look for the "Recruit" button to try again
        b_pts, b_shape, _, _ = find_scaled(gray, btn_temps, args.match_threshold)
        if b_pts:
            human_click(b_pts[0], b_shape)
            # Randomized wait because of bot detection
            time.sleep(args.roll_delay + random.uniform(0, 1))
        else:
            print("Retrying scan...")
            time.sleep(2)

if __name__ == "__main__":
    main()