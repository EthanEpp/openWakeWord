# import numpy as np
# import matplotlib.pyplot as plt
# import mplcursors

# def plot_false_accepts_vs_false_rejects(activation_files, false_activation_file):
#     # Initialize storage for false rejects and thresholds
#     activation_data = {}
#     false_activation_data = {}

#     # Read and process each activation file
#     for label, filepath in activation_files.items():
#         thresholds = []
#         activations = []

#         # Read all lines first
#         with open(filepath, 'r') as f:
#             lines = f.readlines()

#         # Process each line except the last one
#         for line in lines[:-1]:
#             threshold, activation = map(int, line.strip('()\n').split(', '))
#             thresholds.append(threshold)
#             activations.append(activation)

#         # Get total attempts from the last line
#         total_attempts = int(lines[-1].strip())  # The last line contains the total attempts

#         # Calculate false rejects based on the total attempts
#         false_rejects = [(total_attempts - activation) / total_attempts * 100 for activation in activations]

#         activation_data[label] = {
#             'thresholds': np.array(thresholds),
#             'false_rejects': np.array(false_rejects),
#             'activations': np.array(activations)  # Store activations for reference
#         }

#     # Read and process the false activations file
#     thresholds_fp = []
#     false_accepts = []

#     with open(false_activation_file, 'r') as f:
#         for line in f:
#             threshold, false_accept = map(int, line.strip('()\n').split(', '))
#             thresholds_fp.append(threshold)
#             # Convert false accepts to per hour and apply the 0.4 factor
#             false_accepts.append(false_accept * 0.4)

#     false_activation_data = {
#         'thresholds': np.array(thresholds_fp),
#         'false_accepts': np.array(false_accepts)
#     }

#     # Plot the results
#     fig, ax = plt.subplots(figsize=(10, 6))
#     lines = []

#     # Plot the False Accepts vs False Rejects
#     for label, data in activation_data.items():
#         # Interpolate false accepts at the same thresholds as false rejects for consistent plotting
#         false_accepts_interpolated = np.interp(data['thresholds'], false_activation_data['thresholds'], false_activation_data['false_accepts'])
#         line, = ax.plot(false_accepts_interpolated, data['false_rejects'], marker='o', label=label)
#         lines.append(line)

#         # Store the interpolated values in activation_data for hover functionality
#         activation_data[label]['false_accepts_interpolated'] = false_accepts_interpolated

#     # Set up axes
#     ax.set_xlabel('False Accepts per Hour')
#     ax.set_ylabel('False Reject Rate (%)')  # Update label to indicate percentage

#     # Title and grid
#     ax.set_title('False Accepts vs. False Reject Rate')
#     ax.grid(True)

#     # Legend handling
#     ax.legend(lines, [line.get_label() for line in lines], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#     plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the legend

#     # Create a cursor for interactive hovering
#     cursor = mplcursors.cursor(lines, hover=True)

#     # Customize hover annotations to show the interpolated values
#     @cursor.connect("add")
#     def on_add(sel):
#         x = sel.target[0]  # False Accepts (x-axis)
#         y = sel.target[1]  # False Reject Rate (y-axis)

#         annotations = []
#         for label, data in activation_data.items():
#             # Find the index of the closest interpolated false accepts value
#             idx = (np.abs(data['false_accepts_interpolated'] - x)).argmin()
#             threshold = data['thresholds'][idx]
#             false_reject_rate = data['false_rejects'][idx]

#             annotations.append(f"{label}: Threshold = {threshold}, False Reject Rate = {false_reject_rate:.2f}%")

#         # Set the hover annotation text with both the false accepts and the closest matching threshold and reject rate
#         sel.annotation.set_text(f"False Accepts = {x:.2f}\n" + "\n".join(annotations))

#     plt.show()

# # Example usage:
# activation_files = {
#     'No Background': 'examples/audio/snsrdata/no_mask/no_background.txt',
#     'Medium Background': 'examples/audio/snsrdata/no_mask/med_background.txt',
#     'High Background': 'examples/audio/snsrdata/no_mask/high_background.txt',
#     'Max Background': 'examples/audio/snsrdata/no_mask/max_background.txt',
# }
# false_activation_file = 'examples/audio/snsrdata/mask/fpa_hr.txt'

# # plot_from_txt_files(activation_files, false_activation_file)
# plot_false_accepts_vs_false_rejects(activation_files, false_activation_file)
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

def plot_false_accepts_vs_false_rejects(activation_files, false_activation_file):
    # Initialize storage for false rejects and thresholds
    activation_data = {}
    false_activation_data = {}

    # Read and process each activation file
    for label, filepath in activation_files.items():
        thresholds = []
        activations = []

        # Read all lines first
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Process each line except the last one
        for line in lines[:-1]:
            threshold, activation = map(int, line.strip('()\n').split(', '))
            thresholds.append(threshold)
            activations.append(activation)

        # Get total attempts from the last line
        total_attempts = int(lines[-1].strip())  # The last line contains the total attempts

        # Calculate false rejects based on the total attempts
        false_rejects = [(total_attempts - activation) / total_attempts * 100 for activation in activations]

        activation_data[label] = {
            'thresholds': np.array(thresholds),
            'false_rejects': np.array(false_rejects),
            'activations': np.array(activations)  # Store activations for reference
        }

    # Read and process the false activations file
    thresholds_fp = []
    false_accepts = []
    
    # Read all lines from false activation file
    with open(false_activation_file, 'r') as f:
        lines = f.readlines()

    # Get the total number of audio clips from the last line
    total_clips = int(lines[-1].strip())
    
    # Calculate the multiplier (720 / total_clips)
    multiplier = 720 / total_clips
    
    # Process each line except the last one
    for line in lines[:-1]:
        threshold, false_accept = map(int, line.strip('()\n').split(', '))
        thresholds_fp.append(threshold)
        # Convert false accepts to per hour using the calculated multiplier
        false_accepts.append(false_accept * multiplier)

    false_activation_data = {
        'thresholds': np.array(thresholds_fp),
        'false_accepts': np.array(false_accepts)
    }
    # Define a variable for the plot name
    plot_name = 'False Accepts vs. False Reject Rate Sensory No Mask'

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    # Set the window title to the plot name
    fig.canvas.manager.set_window_title(plot_name)
    # Plot the False Accepts vs False Rejects
    for label, data in activation_data.items():
        # Interpolate false accepts at the same thresholds as false rejects for consistent plotting
        false_accepts_interpolated = np.interp(data['thresholds'], false_activation_data['thresholds'], false_activation_data['false_accepts'])
        line, = ax.plot(false_accepts_interpolated, data['false_rejects'], marker='o', label=label)
        lines.append(line)

        # Store the interpolated values in activation_data for hover functionality
        activation_data[label]['false_accepts_interpolated'] = false_accepts_interpolated

    # Set up axes
    ax.set_xlabel('False Accepts per Hour')
    ax.set_ylabel('False Reject Rate (%)')  # Update label to indicate percentage

    # Use the plot_name variable for the title
    ax.set_title(plot_name)
    ax.grid(True)

    # Legend handling
    ax.legend(lines, [line.get_label() for line in lines], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the legend

    # Create a cursor for interactive hovering
    cursor = mplcursors.cursor(lines, hover=True)

    # Customize hover annotations to show the interpolated values
    @cursor.connect("add")
    def on_add(sel):
        x = sel.target[0]  # False Accepts (x-axis)
        y = sel.target[1]  # False Reject Rate (y-axis)

        annotations = []
        for label, data in activation_data.items():
            # Find the index of the closest interpolated false accepts value
            idx = (np.abs(data['false_accepts_interpolated'] - x)).argmin()
            threshold = data['thresholds'][idx]
            false_reject_rate = data['false_rejects'][idx]

            annotations.append(f"{label}: Threshold = {threshold}, False Reject Rate = {false_reject_rate:.2f}%")

        # Set the hover annotation text with both the false accepts and the closest matching threshold and reject rate
        sel.annotation.set_text(f"False Accepts = {x:.2f}\n" + "\n".join(annotations))

    # Print the plot name if needed
    print(f"Generating plot: {plot_name}")

    plt.show()

# Example usage:
activation_files = {
    'No Background': 'examples/audio/snsrdata/no_mask/no_background.txt',
    'Medium Background': 'examples/audio/snsrdata/no_mask/med_background.txt',
    'High Background': 'examples/audio/snsrdata/no_mask/high_background.txt',
    'Max Background': 'examples/audio/snsrdata/no_mask/max_background.txt',
}
false_activation_file = 'examples/audio/snsrdata/no_mask/fpa_hr.txt'

plot_false_accepts_vs_false_rejects(activation_files, false_activation_file)
