
import os
import sys
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cleanup.log")
    ]
)
logger = logging.getLogger(__name__)

def cleanup_stale_files(interval=3600, max_age=3600):
    """
    Clean up stale files in the tmp directory.
    
    Args:
        interval (int): How often to run the cleanup (in seconds). Default 1 hour.
        max_age (int): How old a file must be to be deleted (in seconds). Default 1 hour.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(project_root, "tmp")
    
    logger.info(f"Starting cleanup service. Monitoring {tmp_dir}")
    
    while True:
        try:
            if not os.path.exists(tmp_dir):
                logger.warning(f"Directory {tmp_dir} does not exist. Skipping.")
            else:
                now = time.time()
                count = 0
                for filename in os.listdir(tmp_dir):
                    file_path = os.path.join(tmp_dir, filename)
                    if os.path.isfile(file_path):
                        # Check modification time
                        if now - os.path.getmtime(file_path) > max_age:
                            try:
                                os.remove(file_path)
                                count += 1
                                logger.info(f"Deleted stale file: {filename}")
                            except Exception as e:
                                logger.error(f"Error deleting {filename}: {e}")
                
                if count > 0:
                    logger.info(f"Cleanup cycle completed. Removed {count} files.")
            
            # Wait for next cycle
            time.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
            time.sleep(60) # Retry after 1 minute if critical error

if __name__ == "__main__":
    # Allow command line overrides
    try:
        # Check if run as a one-off script
        if "--once" in sys.argv:
            cleanup_stale_files(interval=0, max_age=3600)
            sys.exit(0)
            
        # Run as daemon
        cleanup_stale_files()
    except KeyboardInterrupt:
        logger.info("Stopping cleanup service.")
