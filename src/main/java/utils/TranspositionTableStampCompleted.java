//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package utils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Transposition Table which can be used when implementing completed UBFM and completed descent.
 * The TT stores the resolution, completion, score and number of visits.
 * Instead of deleting the entire TT (and tree), this implementation only removes the
 * "old" entries, which haven't been seen for the last few searches based on a stamp.
 * <p>
 * Based on implementation from Ludii
 */
public class TranspositionTableStampCompleted extends TranspositionTableStamp {

    //-------------------------------------------------------------------------

    /**
     * Table which stores all Transposition entries
     */
    protected StampTTEntryCompleted[] table;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create a transposition table with number of bits as input
     *
     * @param numBitsPrimaryCode Number of bits used for primary key of TT
     */
    public TranspositionTableStampCompleted(int numBitsPrimaryCode) {
        super(numBitsPrimaryCode);
    }

    /**
     * Creates a new table for all entries
     */
    public void allocate() {
        this.table = new StampTTEntryCompleted[this.maxNumEntries];
    }

    /**
     * Deallocates data with an old stamp, meaning that they haven't been seen for a
     * specified amount of time.
     */
    public void deallocateOldStamps() {
        // For all entries in the table
        StampTTEntryCompleted entry;
        for (int i = 0; i < this.table.length; i++) {
            entry = this.table[i];
            // If it isn't empty
            if (entry != null) {
                // Check all TTData on that entry
                for (int j = entry.data.size() - 1; j >= 0; j--) {
                    // If the data has a "too old" stamp, remove it from the data
                    if (entry.data.get(j).stamp <= this.stamp - this.offSet) {
                        entry.data.remove(j);
                    }
                }
                this.table[i] = entry;
            }
        }
    }

    /**
     * Retreive information from the given hash and update the stamp
     *
     * @param fullHash hash code to retreive
     * @return Data from transposition table, returns null if not available
     */
    @Override
    public StampTTDataCompleted retrieve(long fullHash) {
        StampTTEntryCompleted entry = this.table[(int) (fullHash >>> 64 - this.numBitsPrimaryCode)];
        if (entry != null) {
            Iterator iterator = entry.data.iterator();

            while (iterator.hasNext()) {
                StampTTDataCompleted data = (StampTTDataCompleted) iterator.next();
                if (data.fullHash == fullHash) {
                    // Update stamp and return
                    data.stamp = this.stamp;
                    return data;
                }
            }
        }

        return null;
    }

    /**
     * Stores the given information into the transposition table
     *
     * @param fullHash             hash code to store
     * @param resolution           The resolution (is the game position solved or not)
     * @param completion           The completion (is the game a loss, draw or win? (-1, 0, 1, respectively))
     * @param value                Value which needs to be stored (value found after searching)
     * @param depth                Search depth of hash
     * @param sortedCompletedMoves List with sorted completed moves
     */
    public void store(long fullHash, float resolution, float completion, float value, int depth,
                      List<CompletedMove> sortedCompletedMoves) {
        // Get entry based on primary key
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        StampTTEntryCompleted entry = this.table[idx];

        // If the entry doesn't exist create new entry
        // If the entry exists, but the full hash doesn't add it to the list
        // Otherwise change data of the existing data
        if (entry == null) {
            entry = new StampTTEntryCompleted();
            entry.data.add(new StampTTDataCompleted(fullHash, resolution, completion, value, depth,
                    sortedCompletedMoves, this.stamp));
            this.table[idx] = entry;
        } else {
            StampTTDataCompleted dataToSave = new StampTTDataCompleted(fullHash, resolution, completion, value, depth,
                    sortedCompletedMoves, this.stamp);

            for (int i = 0; i < entry.data.size(); ++i) {
                StampTTData data = (StampTTData) entry.data.get(i);
                if (data.fullHash == fullHash) {
                    entry.data.set(i, dataToSave);
                    return;
                }
            }

            entry.data.add(dataToSave);
        }

    }

    /**
     * Class for all entries in the Transposition Table, existing of TTData
     * <p>
     * Based on implementation from Ludii
     */
    public static final class StampTTEntryCompleted {

        //-------------------------------------------------------------------------

        /**
         * List of Transposition Table Data of a single full hash code
         */
        public List<StampTTDataCompleted> data = new ArrayList(3);

        //-------------------------------------------------------------------------

        /**
         * Constructor for TT entry requiring no inputs
         */
        public StampTTEntryCompleted() {
        }
    }

    /**
     * Transposition Table Data of a single full hash code including a stamp for completed moves
     */
    public static final class StampTTDataCompleted extends StampTTData {

        //-------------------------------------------------------------------------

        /**
         * The resolution (is the game position solved or not)
         */
        public float resolution = 0;

        /**
         * The completion (is the game a loss, draw or win? (-1, 0, 1, respectively))
         */
        public float completion = 0;

        /**
         * Sorted list with all completed moves (all legal moves in the game position)
         */
        public List<CompletedMove> sortedScoredMoves = null;

        //-------------------------------------------------------------------------

        /**
         * Constructor to create the Transposition Table Data with stamp for completed moves
         *
         * @param fullHash       Full hash code
         * @param resolution     The resolution (is the game position solved or not)
         * @param completion     The completion (is the game a loss, draw or win? (-1, 0, 1, respectively))
         * @param value          Value found after searching
         * @param depth          Search depth of the full hash code
         * @param CompletedMoves Sorted list with all completed moves (all legal moves in the game position)
         * @param stamp          Current stamp of the search
         */
        public StampTTDataCompleted(long fullHash, float resolution, float completion, float value,
                                    int depth, List<CompletedMove> CompletedMoves, int stamp) {
            super(fullHash, value, depth, null, stamp);
            this.completion = completion;
            this.resolution = resolution;
            this.sortedScoredMoves = CompletedMoves;
        }
    }
}
