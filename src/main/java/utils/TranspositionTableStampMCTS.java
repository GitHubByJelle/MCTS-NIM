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
 * Transposition Table which can be used when implementing algorithms that don't recognize transpositions.
 * Instead of deleting the entire TT (and tree), this implementation only removes the
 * "old" entries, which haven't been seen for the last few searches based on a stamp.
 *
 * Based on implementation from Ludii
 */
public class TranspositionTableStampMCTS extends TranspositionTableStamp {

    //-------------------------------------------------------------------------

    /** Table which stores all Transposition entries */
    protected StampTTEntryMCTS[] table;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create a transposition table with number of bits as input
     *
     * @param numBitsPrimaryCode Number of bits used for primary key of TT
     */
    public TranspositionTableStampMCTS(int numBitsPrimaryCode) {
        super(numBitsPrimaryCode);
    }

    /**
     * Creates a new table for all entries
     */
    public void allocate() {
        this.table = new StampTTEntryMCTS[this.maxNumEntries];
    }

    /**
     * Deallocates data with an old stamp, meaning that they haven't been seen for a
     * specified amount of time.
     */
    public void deallocateOldStamps() {
        // For all entries in the table
        StampTTEntryMCTS entry;
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
    public StampTTDataMCTS retrieve(long fullHash) {
        StampTTEntryMCTS entry = this.table[(int) (fullHash >>> 64 - this.numBitsPrimaryCode)];
        if (entry != null) {
            Iterator var4 = entry.data.iterator();

            while (var4.hasNext()) {
                StampTTDataMCTS data = (StampTTDataMCTS) var4.next();
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
     * @param fullHash hash code to store
     * @param contextValue Estimated value of context (belonging to hash)
     * @param moveValues Estimated values of all children (belonging to hash)
     */
    public void store(long fullHash, double contextValue, double[] moveValues) {
        // Get entry based on primary key
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        StampTTEntryMCTS entry = this.table[idx];

        // If the entry doesn't exist create new entry
        // If the entry exists, but the full hash doesn't add it to the list
        // Otherwise change data of the existing data
        if (entry == null) {
            entry = new StampTTEntryMCTS();
            entry.data.add(new StampTTDataMCTS(fullHash, contextValue, moveValues, this.stamp));
            this.table[idx] = entry;
        } else {
            StampTTDataMCTS dataToSave = new StampTTDataMCTS(fullHash, contextValue, moveValues,
                    this.stamp);

            for (int i = 0; i < entry.data.size(); ++i) {
                StampTTDataMCTS data = entry.data.get(i);
                if (data.fullHash == fullHash) {
                    entry.data.set(i, dataToSave);
                    return;
                }
            }

            entry.data.add(dataToSave);
        }
    }

    /**
     * Stores the given information into the transposition table. It only overwrites the contextValue, but keeps
     * the move values untouched.
     *
     * @param fullHash hash code to store
     * @param contextValue Estimated value of context (belonging to hash)
     */
    public void storeContextValue(long fullHash, double contextValue) {
        // Get entry based on primary key
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        StampTTEntryMCTS entry = this.table[idx];

        // If the entry doesn't exist create new entry
        // If the entry exists, but the full hash doesn't add it to the list
        // Otherwise change data of the existing data
        if (entry == null) {
            entry = new StampTTEntryMCTS();
            entry.data.add(new StampTTDataMCTS(fullHash, contextValue, null, this.stamp));
            this.table[idx] = entry;
        } else {
            for (int i = 0; i < entry.data.size(); ++i) {
                StampTTDataMCTS data = entry.data.get(i);
                if (data.fullHash == fullHash) {
                    entry.data.get(i).contextValue = contextValue;
                    return;
                }
            }

            StampTTDataMCTS dataToSave = new StampTTDataMCTS(fullHash, contextValue, null,
                    this.stamp);
            entry.data.add(dataToSave);
        }
    }

    /**
     * Stores the given information into the transposition table. It only overwrites the moveValues, but keeps
     * the contextValue untouched.
     *
     * @param fullHash hash code to store
     * @param moveValues Estimated values of all children (belonging to hash)
     */
    public void storeMoveValues(long fullHash, double[] moveValues) {
        // Get entry based on primary key
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        StampTTEntryMCTS entry = this.table[idx];

        // If the entry doesn't exist create new entry
        // If the entry exists, but the full hash doesn't add it to the list
        // Otherwise change data of the existing data
        if (entry == null) {
            entry = new StampTTEntryMCTS();
            entry.data.add(new StampTTDataMCTS(fullHash, -Value.INF, moveValues, this.stamp));
            this.table[idx] = entry;
        } else {
            for (int i = 0; i < entry.data.size(); ++i) {
                StampTTDataMCTS data = entry.data.get(i);
                if (data.fullHash == fullHash) {
                    entry.data.get(i).moveValues = moveValues;
                    return;
                }
            }

            StampTTDataMCTS dataToSave = new StampTTDataMCTS(fullHash, -Value.INF, moveValues,
                    this.stamp);
            entry.data.add(dataToSave);
        }
    }



    /**
     * Class for all entries in the Transposition Table, existing of TTData
     *
     * Based on implementation from Ludii
     */
    public static final class StampTTEntryMCTS {

        //-------------------------------------------------------------------------

        /** List of Transposition Table Data of a single full hash code */
        public List<StampTTDataMCTS> data = new ArrayList(3);

        //-------------------------------------------------------------------------

        /**
         * Constructor for TT entry requiring no inputs
         */
        public StampTTEntryMCTS() {
        }
    }

    /**
     * Transposition Table Data of a single full hash code including a stamp
     */
    public static class StampTTDataMCTS extends StampTTData {

        //-------------------------------------------------------------------------

        /** Best heuristic value from all children */
        public double contextValue;

        /** Initial heuristic value (before minimax changes them) */
        public double[] moveValues;

        //-------------------------------------------------------------------------

        /**
         * Constructor to create the Transposition Table Data with stamp an evaluator which doesn't recognize
         * transpositions.
         *
         * @param fullHash Full hash code
         * @param contextValue Estimated value of context (belonging to hash)
         * @param moveValues Estimated values of all children (belonging to hash)
         * @param stamp Current stamp of the search
         */
        public StampTTDataMCTS(long fullHash, double contextValue,
                               double[] moveValues, int stamp) {
            super(fullHash, 0f, 0, null, stamp);
            this.contextValue = contextValue;
            this.moveValues = moveValues;
        }
    }
}
