import logging
import time

# ... existing code ...

def compile_model(model, **kwargs):
    start_time = time.time()
    logging.info("Compilation started")
    
    # ... existing code ...
    
    # Add a progress log every 30 seconds
    last_log_time = start_time
    while not compilation_finished:
        # ... existing code ...
        current_time = time.time()
        if current_time - last_log_time >= 30:
            logging.info("Compilation in progress... ({} seconds elapsed)".format(int(current_time - start_time)))
            last_log_time = current_time
    
    logging.info("Compilation finished ({} seconds elapsed)".format(int(time.time() - start_time)))
    return compiled_model

# ... existing code ...