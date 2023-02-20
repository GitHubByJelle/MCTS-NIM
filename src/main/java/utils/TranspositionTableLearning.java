package utils;

//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Transposition Table which can be used to store data for learning (training a NN).
 *
 * Based on implementation from Ludii
 */
public class TranspositionTableLearning {

    //-------------------------------------------------------------------------

    /** Number of bits used for primary key */
    private final int numBitsPrimaryCode;

    /** Number of entries in the TT (size of TT is 2^number of bits)*/
    private final int maxNumEntries;

    /** Table which stores all Transposition entries */
    private learningTTEntry[] table;

    //-------------------------------------------------------------------------

    /**
     * Constructor to create a transposition table with number of bits as input
     *
     * @param numBitsPrimaryCode Number of bits used for primary key of TT
     */
    public TranspositionTableLearning(int numBitsPrimaryCode) {
        this.numBitsPrimaryCode = numBitsPrimaryCode;
        this.maxNumEntries = 1 << numBitsPrimaryCode;
        this.table = null;
    }

    /**
     * Creates a new table for all entries
     */
    public void allocate() {
        this.table = new learningTTEntry[this.maxNumEntries];
    }

    /**
     * Removes the entire table from the memory
     */
    public void deallocate() {
        this.table = null;
    }

    /**
     * Checks if a table is allocated
     * @return True if a tables is allocated, false otherwise
     */
    public boolean isAllocated() {
        return this.table != null;
    }

    /**
     * Retreive information from the given hash
     *
     * @param fullHash hash code to retreive
     * @return Data from transposition table, returns null if not available
     */
    public learningTTData retrieve(long fullHash) {
        // Get entry based on primary key
        learningTTEntry entry = this.table[(int) (fullHash >>> 64 - this.numBitsPrimaryCode)];

        // If entry exist, check which object in the list matches the hash
        if (entry != null) {
            Iterator iterator = entry.data.iterator();

            while (iterator.hasNext()) {
                learningTTData data = (learningTTData) iterator.next();
                if (data.fullHash == fullHash) {
                    return data;
                }
            }
        }

        // Return null if nothing exists
        return null;
    }

    /**
     * Stores the given information into the transposition table
     *
     * @param fullHash hash code to store
     * @param value Value which needs to be stored (value found after searching)
     * @param depth Search depth of hash
     * @param inputNN The input for the NN representing the game position of the hash
     */
    public void store(long fullHash, float value, int depth, float[] inputNN) {
        // Get entry based on primary key
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        learningTTEntry entry = this.table[idx];

        // If the entry doesn't exist create new entry
        // If the entry exists, but the full hash doesn't add it to the list
        // Otherwise change data of the existing data
        if (entry == null) {
            entry = new learningTTEntry();
            entry.data.add(new learningTTData(fullHash, value, depth, inputNN));
            this.table[idx] = entry;
        } else {
            learningTTData dataToSave = new learningTTData(fullHash, value, depth, inputNN);

            for (int i = 0; i < entry.data.size(); ++i) {
                learningTTData data = (learningTTData) entry.data.get(i);
                if (data.fullHash == fullHash) {
                    entry.data.set(i, dataToSave);
                    return;
                }
            }

            entry.data.add(dataToSave);
        }

    }

    /**
     * Getter for the table of entries
     * @return table of entries
     */
    public learningTTEntry[] getTable() {
        return this.table;
    }

    /**
     * Count number of entries in the Transposition Table
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
     * Returns the correct entry based on the hash code
     *
     * @param fullHash hash code of game position
     * @return Entry belonging to the game position
     */
    public learningTTEntry getEntry(long fullHash){
        int idx = (int) (fullHash >>> 64 - this.numBitsPrimaryCode);
        return this.table[idx];
    }

    /**
     * Class for all entries in the Transposition Table, existing of TTData
     *
     * Based on implementation from Ludii
     */
    public static final class learningTTEntry {

        //-------------------------------------------------------------------------

        /** List of Transposition Table Data of a single full hash code */
        public List<learningTTData> data = new ArrayList(3);

        //-------------------------------------------------------------------------

        /**
         * Constructor for TT entry requiring no inputs
         */
        public learningTTEntry() {
        }
    }

    /**
     * Transposition Table Data of a single full hash code
     */
    public static final class learningTTData {

        //-------------------------------------------------------------------------

        /** Full hash code */
        public long fullHash = -1L;

        /** Value found after searching */
        public float value = Float.NaN;

        /** Search depth of the full hash code */
        public int depth = -1;

        /** The input for the NN representing the game position of the hash */
        public float[] inputNN = null;

        //-------------------------------------------------------------------------

        /**
         * Constructor to create the Transposition Table Data
         *
         * @param fullHash Full hash code
         * @param value Value found after searching
         * @param depth Search depth of the full hash code
         * @param inputNN The input for the NN representing the game position of the hash
         */
        public learningTTData(long fullHash, float value, int depth, float[] inputNN) {
            this.fullHash = fullHash;
            this.value = value;
            this.depth = depth;
            this.inputNN = inputNN;
        }
    }
}
