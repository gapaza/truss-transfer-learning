package rbs.transformer;

import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.core.*;
import org.moeaframework.problem.AbstractProblem;
import com.mathworks.engine.*;
import java.util.concurrent.*;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryVariable;
import java.util.ArrayList;


// import seakers.trussaos.problems.GabeTrussProblem;
// import seakers.trussaos.problems.GabeTrussProblemFibre;
// import seakers.trussaos.problems.GabeArteryProblem;

import seakers.trussaos.problems.TrussProblem;


public class EvaluationTruss {

    public TrussProblem problem;
    public MatlabEngine engine = null;
    public boolean debug = false;
    public int eval_count = 0;

    public EvaluationTruss(){

        // ------------------------------
        // MATLAB Engine
        // ------------------------------
        this.newMatlabEngine();

        // ------------------------------
        // TrussProblem
        // ------------------------------
        this.problem = new TrussProblem(this.engine);

    }

    // ------------------------------
    // Evaluate Design
    // ------------------------------

    public ArrayList<Double> evaluateDesign(
        ArrayList<Double> design_double,
        int problem_num,
        double sidenum,
        double member_radius,
        double member_length,
        double y_modulus,
        boolean calc_constraints,
        boolean calc_heuristics
    ){
        ArrayList<Integer> design = this.convertDesign(design_double);
        ArrayList<Double> results = this.problem.evaluate(
            design,
            problem_num,
            sidenum,
            member_length,
            member_radius,
            y_modulus,
            calc_heuristics,
            calc_constraints
        );

        if(this.debug == true){
            // 4. Print results
            System.out.println("----- Results -----");
            System.out.println("Horizontal Stiffness: " + results.get(0));
            System.out.println("Vertical Stiffness: " + results.get(1));
            System.out.println("Stiffness Ratio: " + results.get(2));
            System.out.println("Volume Fraction: " + results.get(3));

            if (calc_constraints) {
                System.out.println("\n----- Constraints -----");
                System.out.println("Feasibility Constraint: " + results.get(4));
                System.out.println("Connectivity Constraint: " + results.get(5));
            }

            if (calc_heuristics) {
                System.out.println("\n----- Heuristics -----");
                System.out.println("Collapsability Heuristic: " + results.get(6));
                System.out.println("Connectivity Heuristic: " + results.get(7));
                System.out.println("Orientation Heuristic: " + results.get(8));
                System.out.println("Intersection Heuristic: " + results.get(9));
            }
        }
        this.eval_count += 1;
        if(this.eval_count > 10000){
            this.refreshEngine();
            this.eval_count = 0;
        }
        return results;
    }

    // ------------------------------
    // Refresh Engine
    // ------------------------------
    public void refreshEngine(){
        this.newMatlabEngine();
        this.problem = new TrussProblem(this.engine);
    }

    // ------------------------------
    // MATLAB Engine
    // ------------------------------

    public void newMatlabEngine(){
        try {
            if (this.engine != null) {
                System.out.println("Closing MATLAB engine...");
                this.engine.close();
                System.out.println("MATLAB engine closed.");
            }
            System.out.println("Starting MATLAB engine...");
            String myPath = "/Users/gapaza/repos/seakers/truss/KDDMM/Truss_AOS";
            this.engine = MatlabEngine.startMatlab();
            this.engine.eval("addpath('" + myPath + "')", null, null);
            this.engine.eval("warning('off', 'MATLAB:singularMatrix');");
            System.out.println("MATLAB engine started.");
        }
        catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
    }

    // ------------------------------
    // Helpers
    // ------------------------------

    public ArrayList<Integer> convertDesign(ArrayList<Double> design){
        ArrayList<Integer> design_int = new ArrayList<Integer>();
        for (int i = 0; i < design.size(); i++) {
            design_int.add(design.get(i).intValue());
        }
        return design_int;
    }
}