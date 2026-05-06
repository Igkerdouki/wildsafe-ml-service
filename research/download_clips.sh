#!/bin/bash
# Download wildlife test clips for model evaluation

BASE_DIR="/Users/ioannagkerdouki/wildsafe-ml-service/research/test_data"

download_clip() {
    local animal="$1"
    local url="$2"
    local start="$3"
    local end="$4"
    local desc="$5"
    local filename="${animal}_$(echo "$desc" | tr ' ' '_' | tr -cd '[:alnum:]_').mp4"

    echo "Downloading: $animal - $desc"
    yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --download-sections "*${start}-${end}" \
        --force-keyframes-at-cuts \
        -o "${BASE_DIR}/${animal}/${filename}" \
        --no-playlist \
        --quiet --progress \
        "$url" 2>/dev/null || echo "  Failed: $url"
}

# Deer
download_clip "deer" "https://youtu.be/FaPPUQOoTr4" "2:33" "2:40" "eating_head_down"
download_clip "deer" "https://youtu.be/bWrNWRKLiTI" "0:16" "0:23" "two_deer_staring"
download_clip "deer" "https://youtu.be/G8X594TIiag" "7:03" "7:09" "multiple_eating_grass"
download_clip "deer" "https://youtu.be/O1ogQr87VJI" "9:31" "9:33" "walking_through_grass"

# Raccoon
download_clip "raccoon" "https://youtu.be/b6fA_Y8bbHE" "0:20" "0:24" "on_boat_with_person"
download_clip "raccoon" "https://youtu.be/T9DYbX4dvpI" "0:25" "0:28" "sitting_in_wheel"
download_clip "raccoon" "https://youtu.be/X7JSzAJ3vYs" "4:41" "4:45" "moving_eating_food"
download_clip "raccoon" "https://youtu.be/MFbkUAgxK4Q" "1:35" "1:40" "still_on_road"

# Fox
download_clip "fox" "https://youtu.be/CQHTth6AQNU" "0:40" "0:49" "sleepy_on_grass"
download_clip "fox" "https://youtu.be/CQHTth6AQNU" "2:01" "2:06" "sitting_on_road"
download_clip "fox" "https://youtu.be/SLVXapC_X84" "3:03" "3:08" "being_pet_grass"

# Coyote
download_clip "coyote" "https://youtu.be/AYCSM_Gdu4c" "1:28" "1:31" "running_on_grass"
download_clip "coyote" "https://www.youtube.com/shorts/zdN2H9Grigg" "0:00" "0:12" "barking_on_rock"
download_clip "coyote" "https://youtu.be/ObFuD6Opb5o" "2:22" "2:26" "yawning"

# Opossum
download_clip "opossum" "https://youtu.be/bRIVTEBaLkA" "0:00" "0:10" "eating_bananas"
download_clip "opossum" "https://youtu.be/PtVG6DHOA1o" "3:33" "3:41" "eating_from_person"
download_clip "opossum" "https://youtu.be/GLQr1wLr_Xo" "0:48" "1:00" "walking_through_woods"

# Skunk
download_clip "skunk" "https://youtu.be/Ttn5iQZ-4XI" "2:17" "2:23" "eating_from_bowl"
download_clip "skunk" "https://youtu.be/vszWsYOdnPg" "0:05" "0:14" "group_towards_camera"
download_clip "skunk" "https://youtu.be/5RiErxwrVQ0" "0:54" "1:07" "looking_around"

# Bear
download_clip "bear" "https://youtu.be/aAfXsxLSblM" "16:09" "16:11" "walking_woods"
download_clip "bear" "https://youtu.be/aAfXsxLSblM" "40:33" "40:38" "walking_beach"
download_clip "bear" "https://youtu.be/J184FmCiuLk" "1:15" "1:22" "two_bears_with_human"

# Elk
download_clip "elk" "https://youtu.be/9TW8Gf3kF4Q" "0:00" "0:11" "walking_field_1"
download_clip "elk" "https://youtu.be/9TW8Gf3kF4Q" "0:12" "0:22" "walking_field_2"
download_clip "elk" "https://youtu.be/PqYEN_NKJk8" "1:21" "1:27" "walking_field_3"

# Moose
download_clip "moose" "https://youtu.be/F3yse7vTWrw" "6:28" "6:31" "walking_field_1"
download_clip "moose" "https://youtu.be/rITjvTwU_Tw" "4:42" "4:53" "walking_field_2"
download_clip "moose" "https://youtu.be/7PMO1uQrmGg" "0:43" "0:51" "walking_field_3"

# Goat
download_clip "goat" "https://youtu.be/RG9TMn1FJzc" "0:42" "0:47" "walking_field"
download_clip "goat" "https://youtu.be/RG9TMn1FJzc" "1:16" "1:24" "walking_mountain"
download_clip "goat" "https://youtu.be/Dercp0lOI0Y" "0:00" "0:20" "licking_arm"

# Horse
download_clip "horse" "https://youtu.be/bHiZjBCnGbM" "0:13" "0:17" "walking_field"
download_clip "horse" "https://youtu.be/bHiZjBCnGbM" "15:12" "15:20" "eating_grass"
download_clip "horse" "https://youtu.be/bHiZjBCnGbM" "43:46" "43:57" "horses_walking"

# Wild Boar
download_clip "wild_boar" "https://youtu.be/Fgik_tTShzE" "5:15" "5:20" "eating_wild"
download_clip "wild_boar" "https://youtu.be/Fgik_tTShzE" "8:43" "8:47" "seen_wild"
download_clip "wild_boar" "https://youtu.be/Fgik_tTShzE" "29:42" "29:47" "running_forest"

echo ""
echo "Download complete! Checking results..."
find "$BASE_DIR" -name "*.mp4" -exec du -h {} \; | sort -k2
echo ""
echo "Total size:"
du -sh "$BASE_DIR"
