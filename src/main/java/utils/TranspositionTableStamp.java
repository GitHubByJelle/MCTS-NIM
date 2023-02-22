//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package utils;

import utils.data_structures.ScoredMove;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Transposition Table which can be used when implementing Best-First search implementations.
 * Instead of deleting the entire TT (and tree), this implementation only removes the
 * "old" entries, which haven't been seen for the last few searches based on a stamp.
 * <p>
 * Based on implementation from Ludii
 */
public class TranspositionTableStamp {

    //-------------------------------------------------------------------------

    /**
     * Number of bits used for primary key
     */
    protected final int numBitsPrimaryCode;

    /**
     * Number of entries in the TT (size of TT is 2^number of bits)
     */
    protected final int maxNumEntries;

    /**
     * Table which stores all Transposition entries
     */
    private StampTTEntry[] table;

    /**
     * Current stamp that keeps track of the visits of the data
     */
    protected int stamp;

    /**
     * Number of games the data will be kept after being seen for the last time
     */
    protected int offSet = 3;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create a transposition table with number of bits as input
     *
     * @param numBitsPrimaryCode Number of bits used for primary key of TT
     */
    public TranspositionTableStamp(int numBitsPrimaryCode) {
        this.numBitsPrimaryCode = numBitsPrimaryCode;
        this.maxNumEntries = 1 << numBitsPrimaryCode;
        this.table = null;
        this.stamp = 0;
    }

    /**
     * Creates a new table for all entries
     */
    public void allocate() {
        this.table = new StampTTEntry[this.maxNumEntries];
    }

    /**
     * Removes the entire table from the memory
     */
    public void deallocate() {
        this.table = null;
    }

    /**
     * Checks if a table is allocated
     *
     * @return True if a tables is allocated, false otherwise
     */
    public boolean isAllocated() {
        return this.table != null;
    }

    /**
     * Deallocates data with an old stamp, meaning that they haven't been seen for a
     * specified amount of time.
     */
    public void deallocateOldStamps() {
        // For all entries in the table
        StampTTEntry entry;
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
     * Checks all entries and prints the oldest and newest stamp in the Transposition Table.
     */
    public void stampCheck() {
        // For all entires
        StampTTEntry entry;
        int maxStamp = -1;
        int minStamp = 9999999;
        for (int i = 0; i < this.table.length; i++) {
            entry = this.table[i];
            // If it isn't empty
            if (entry != null) {
                // Check all TTData on that entry
                for (int j = 0; j < entry.data.size(); j++) {
                    // If the stamp is more recent, save
                    if (entry.data.get(j).stamp > maxStamp) {
                        maxStamp = entry.data.get(j).stamp;
                    }
                    // If the stamp is older, save
                    if (entry.data.get(j).stamp < minStamp) {
                        minStamp = entry.data.get(j).stamp;
                    }
                }
            }
        }

        // Print results
        System.out.println("Maximum stamp: " + maxStamp + ", Minimum stamp: " + minStamp + ".");
    }

    /**
     * Updates the current stamp of the TT. Needs to be performed when the search has finished.
     */
    public void updateStamp() {
        this.stamp += 1;
    }

    /**
     * Resets the stamp back to 0
     */
    public void resetStamp() {
        this.stamp = 0;
    }

    /**
     * Retreive information from the given hash and update the stamp
     *
     * @param fullHash hash code to retreive
     * @return Data from transposition table, returns null if not available
     */
    public StampTTData retrieve(long fullHash) {
        StampTTEntry entry = this.table[(int) (fullHash >>> 64 - this.numBitsPrimaryCode)];
        if (entry != null) {
            Iterator iterator = entry.data.iterator();

            while (iterator.hasNext()) {
                StampTTData data = (StampTTData) iterator.next();
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
     * @param fullHash          hash code to store
     * @param value             Value which needs to be stored (value found after searching)
     * @param depth             Search depth of hash
     * @param sortedScoredMoves List with sorted scored moves
     */
    public void store(long fullHash, float value, int depth, List<ScoredMove> sortedScoredMoves) {
        // Get entry based on primary key
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        StampTTEntry entry = this.table[idx];

        // If the entry doesn't exist create new entry
        // If the entry exists, but the full hash doesn't add it to the list
        // Otherwise change data of the existing data
        if (entry == null) {
            entry = new StampTTEntry();
            entry.data.add(new StampTTData(fullHash, value, depth, sortedScoredMoves, this.stamp));
            this.table[idx] = entry;
        } else {
            StampTTData dataToSave = new StampTTData(fullHash, value, depth, sortedScoredMoves, this.stamp);

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
     * Count number of entries in the Transposition Table
     *
     * @return the number of entries
     */
    public int nbEntries() {
        int res = 0;

        for (int i = 0; i < this.maxNumEntries; ++i) {
            if (this.table[i] != null) {
                res += this.table[i].data.size();
            }
        }

        return res;
    }

    /**
     * Class for all entries in the Transposition Table, existing of TTData
     * <p>
     * Based on implementation from Ludii
     */
    public static final class StampTTEntry {

        //-------------------------------------------------------------------------

        /**
         * List of Transposition Table Data of a single full hash code
         */
        public List<StampTTData> data = new ArrayList(3);

        //-------------------------------------------------------------------------

        /**
         * Constructor for TT entry requiring no inputs
         */
        public StampTTEntry() {
        }
    }

    /**
     * Transposition Table Data of a single full hash code including a stamp
     */
    public static class StampTTData {

        //-------------------------------------------------------------------------

        /**
         * Full hash code
         */
        public long fullHash = -1L;

        /**
         * Value found after searching
         */
        public float value = Float.NaN;

        /**
         * Search depth of the full hash code
         */
        public int depth = -1;

        /**
         * Sorted list with all scored moves (all legal moves in the game position)
         */
        public List<ScoredMove> sortedScoredMoves = null;

        /**
         * Current stamp of the search
         */
        protected int stamp;

        /**
         * Constructor to create the Transposition Table Data with stamp
         *
         * @param fullHash          Full hash code
         * @param value             Value found after searching
         * @param depth             Search depth of the full hash code
         * @param sortedScoredMoves Sorted list with all scored moves (all legal moves in the game position)
         * @param stamp             Current stamp of the search
         */
        public StampTTData(long fullHash, float value, int depth, List<ScoredMove> sortedScoredMoves, int stamp) {
            this.fullHash = fullHash;
            this.value = value;
            this.depth = depth;
            this.sortedScoredMoves = sortedScoredMoves;
            this.stamp = stamp;
        }
    }
}
