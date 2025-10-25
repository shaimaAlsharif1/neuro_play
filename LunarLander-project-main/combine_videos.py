import os
from moviepy import VideoFileClip, concatenate_videoclips

base_dir = "videos"
categories = ["train_video", "eval_video", "random_video"]

for category in categories:
    category_path = os.path.join(base_dir, category)
    if not os.path.exists(category_path):
        print(f"⚠️ Skipping {category_path}, not found.")
        continue

    print(f"🎬 Processing category: {category}")
    video_files = sorted(
        [f for f in os.listdir(category_path) if f.endswith(".mp4")]
    )

    if not video_files:
        print(f"⚠️ No videos found in {category_path}")
        continue

    clips = []
    for filename in video_files:
        video_path = os.path.join(category_path, filename)
        clips.append(VideoFileClip(video_path))

    final = concatenate_videoclips(clips, method="compose")

    # Ensure output directory exists
    output_dir = os.path.join(base_dir, "combined")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"combined_{category}.mp4")

    final.write_videofile(output_path, codec="libx264", audio=False)
    print(f"✅ Combined video saved: {output_path}")






# import os
# from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip



# base_dir = "videos"
# categories = {
#     "train_video": "🚀 Training Phase",
#     "eval_video": "🧠 Evaluation Phase",
#     "random_video": "🎲 Random Agent"
# }

# for category, label in categories.items():
#     category_path = os.path.join(base_dir, category)
#     if not os.path.exists(category_path):
#         print(f"⚠️ Skipping {category_path}, not found.")
#         continue

#     print(f"🎬 Processing category: {category}")
#     video_files = sorted(
#         [f for f in os.listdir(category_path) if f.endswith(".mp4")]
#     )

#     if not video_files:
#         print(f"⚠️ No videos found in {category_path}")
#         continue

#     clips = []

#     # Add a title card for each category (3 seconds)

#     txt_clip = TextClip(
#         label,             # first positional argument = text
#         color="white",     # color keyword
#         fontsize=50,       # fontsize keyword
#         size=(1280, 720),  # size keyword
#         method="caption"   # method keyword
#     )
#     txt_clip = txt_clip.set_duration(3).on_color(
#         size=(1280, 720), color=(0, 0, 0), col_opacity=1
#     ).set_position("center")


#     clips.append(txt_clip)

#     # Add actual videos
#     for filename in video_files:
#         video_path = os.path.join(category_path, filename)
#         clips.append(VideoFileClip(video_path))

#     final = concatenate_videoclips(clips, method="compose")

#     output_dir = os.path.join(base_dir, "combined")
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"combined_{category}.mp4")

#     final.write_videofile(output_path, codec="libx264", audio=False)
#     print(f"✅ Combined video saved: {output_path}")
