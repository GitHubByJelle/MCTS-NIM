package Evaluator;

import game.Game;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import other.context.Context;
import utils.TranspositionTableStampMCTS;
import utils.Value;

import java.util.ArrayList;

public class MultiNeuralNetworkTTLeafEvaluator extends MultiNeuralNetworkLeafEvaluator {

    //-------------------------------------------------------------------------

    TranspositionTableStampMCTS TT = new TranspositionTableStampMCTS(20);

    //-------------------------------------------------------------------------

    public MultiNeuralNetworkTTLeafEvaluator(Game game, MultiLayerNetwork net, int nThreads) {
        super(game, net, nThreads);
        this.TT.allocate();
    }

    public float evaluate(Context context, int maximisingPlayer) {
        long zobrist = context.state().fullHash(context);
        TranspositionTableStampMCTS.StampTTDataMCTS TTData = this.TT.retrieve(zobrist);
        float value;
        if (TTData == null || TTData.contextValue == -Value.INF){
            value = super.evaluate(context, maximisingPlayer);
            if (TTData == null){
                this.TT.storeContextValue(zobrist, value);
            }
            else {
                TTData.contextValue = value;
            }
        }
        else {
            value = (float)TTData.contextValue;
        }

        return value;
    }

    public float[] evaluateMoves(Context context, ArrayList<Integer> nonTerminalMoves, int maximisingPlayer) {
        long zobrist = context.state().fullHash(context);
        TranspositionTableStampMCTS.StampTTDataMCTS TTData = this.TT.retrieve(zobrist);
        float[] values;
        if (TTData == null || TTData.contextValue == -Value.INF){
            values = super.evaluateMoves(context, nonTerminalMoves, maximisingPlayer);
            if (TTData == null){
                this.TT.storeMoveValues(zobrist, this.convertFloatsToDoubles(values));
            }
            else {
                TTData.moveValues = this.convertFloatsToDoubles(values);
            }
        }
        else {
            values = this.convertDoublesToFloats(TTData.moveValues);
        }

        return values;
    }

    public double[] convertFloatsToDoubles(float[] input)
    {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = input[i];
        }
        return output;
    }

    public float[] convertDoublesToFloats(double[] input)
    {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = (float)input[i];
        }
        return output;
    }
}
