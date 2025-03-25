import sys
from datetime import datetime
from pathlib import Path

# Add SDK path
sdk_path = Path(__file__).resolve().parent / 'FitSDKRelease_21.158.00' / 'py'
sys.path.insert(0, str(sdk_path))

# Now import the SDK components
from garmin_fit_sdk import fit, profile

# Create a new FIT file encoder
encoder = fit.Encoder()

# Start the FIT file with a File ID message
file_id_msg = fit.FileIdMesg()
file_id_msg.set_type(profile.File.WORKOUT)
file_id_msg.set_manufacturer(profile.Manufacturer.DEVELOPMENT)
file_id_msg.set_time_created(datetime.utcnow())
encoder.write(file_id_msg)

# Create a Workout message
workout_msg = fit.WorkoutMesg()
workout_msg.set_sport(profile.Sport.RUNNING)
workout_msg.set_num_valid_steps(2)
encoder.write(workout_msg)

# Step 1 - Warm up for 10 minutes
step1 = fit.WorkoutStepMesg()
step1.set_message_index(0)
step1.set_duration_type(profile.WorkoutStepDuration.TIME)
step1.set_duration_value(600)  # 600 seconds
step1.set_intensity(profile.Intensity.WARMUP)
step1.set_notes("Warm up")
encoder.write(step1)

# Step 2 - Zone 2 run for 15 minutes
step2 = fit.WorkoutStepMesg()
step2.set_message_index(1)
step2.set_duration_type(profile.WorkoutStepDuration.TIME)
step2.set_duration_value(900)
step2.set_intensity(profile.Intensity.ACTIVE)
step2.set_target_type(profile.WorkoutStepTarget.HEART_RATE)
step2.set_target_value(2)  # HR zone 2
step2.set_notes("Zone 2 Run")
encoder.write(step2)

# Finalize and write to file
output_file = "test_workout.fit"
with open(output_file, "wb") as f:
    encoder.finish(f)

print(f"✅ Workout FIT file generated: {output_file}")
