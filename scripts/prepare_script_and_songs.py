import torch
from utils.script_and_song import ScriptPlanner, SongPreparer

if __name__ == "__main__":
    planner = ScriptPlanner()
    scenes = planner.draft_script("A girl fighting for justice in society.", num_scenes=12)
    song_cues = planner.prep_song_cues(num_songs=2)
    print("--- Script Scenes ---")
    for scene in scenes:
        print(scene)
    print("--- Song Cues ---")
    for cue in song_cues:
        print(cue)

    songgen = SongPreparer()
    for cue in song_cues:
        context = cue.split(':')[-1].strip()
        print(songgen.generate_song(context))
