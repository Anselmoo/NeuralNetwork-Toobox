"""
analytic_generator.py is a tool for generating logic-tables for random logic-operators
as csv-files for machine learning.
"""
import numpy as np
import itertools
import hashlib


class LogicOperators:
    """LogicOperators.
    
    Attributes
    ----------
    logic operators functions:
        1. lg_and
        2. lg_or
        3. lg_xor
        4. lg_not_and
        5. lg_not_or
    call functions for the logic operators

    """

    @staticmethod
    def lg_and(x1, x2, dtype=np.int):
        """lg_and.

        Parameters
        ----------
        x1 :
            x1
        x2 :
            x2
        dtype :
            dtype
        """
        return np.logical_and(x1, x2, dtype=dtype)

    @staticmethod
    def lg_or(x1, x2, dtype=np.int):
        """lg_or.

        Parameters
        ----------
        x1 :
            x1
        x2 :
            x2
        dtype :
            dtype
        """
        return np.logical_or(x1, x2, dtype=dtype)

    @staticmethod
    def lg_xor(x1, x2, dtype=np.int):
        """lg_xor.

        Parameters
        ----------
        x1 :
            x1
        x2 :
            x2
        dtype :
            dtype
        """
        return np.logical_xor(x1, x2, dtype=dtype)

    @staticmethod
    def lg_not_and(x1, x2, dtype=np.int):
        """lg_not_and.

        Parameters
        ----------
        x1 :
            x1
        x2 :
            x2
        dtype :
            dtype
        """
        return np.logical_not(np.logical_and(x1, x2, dtype=dtype))

    @staticmethod
    def lg_not_or(x1, x2, dtype=np.int):
        """lg_not_or.

        Parameters
        ----------
        x1 :
            x1
        x2 :
            x2
        dtype :
            dtype
        """
        return np.logical_not(np.logical_or(x1, x2, dtype=dtype))

    @staticmethod
    def get_operator(lgopt, x1, x2):
        """Get the logic operators.

        get_operator is calling the static logic functions from above:
        
        1. lg_and
        2. lg_or
        3. lg_xor
        4. lg_not_and
        5. lg_not_or

        Parameters
        ----------
        lgopt : str
            Initial str for calling the logic operators
        x1 : int or bool as array-like
            x1, first value of the logic operator
        x2 : int or bool as array-like
            x2, second value of the logic operator

        Return
        ------
         : bool
            Returns the bool single or bool array depending on the logic operators
        """
        if lgopt.lower() == "and":
            return LogicOperators.lg_and(x1, x2)
        elif lgopt.lower() == "or":
            return LogicOperators.lg_or(x1, x2)
        elif lgopt.lower() == "xor":
            return LogicOperators.lg_xor(x1, x2)
        elif lgopt.lower() == "nand":
            return LogicOperators.lg_not_and(x1, x2)
        elif lgopt.lower() == "nor":
            return LogicOperators.lg_not_or(x1, x2)


class LogicGenerator(LogicOperators, object):
    """LogicGenerator.
    """

    __lst = [0, 1]
    # only input parameters
    logic_items = 0
    # optional input
    fname = None
    # internal variables non-privat
    logic_variables = 0
    logic_init_mat = np.array([])
    logic_result = np.array([])
    logic_operators = []
    # export parameters
    logic_res_mat = np.array([])
    logic_expression = ""

    def __init__(self, logic_items=5, fname=None):
        """__init__.

        Parameters
        ----------
        logic_items : int, optional
            Total number of logical operators, by default 5.
        fname : str, optional
            Filename to save as txt, by default None
        """
        super().__init__()

        self.logic_items = logic_items
        self.logic_variables = self.logic_items + 1
        self.logic_operations()
        self.fname = fname

    def logic_operations(self):
        """Generate a random connections of logic opperators."""
        self.logic_init_mat = self.logic_matrices()
        ref_list = ["and", "or", "xor", "nand", "nor"]
        rnd_range = np.random.randint(len(ref_list), size=self.logic_items)
        
        for i in rnd_range:
            self.logic_operators.append(ref_list[i])

    def logic_matrices(self):
        """Generate a logic matrice with all possible combinations."""
        logic_mat = np.asarray(
            list(itertools.product(self.__lst, repeat=self.logic_variables))
        )
        return logic_mat

    def generator(self):
        """generator.
        """

        self.logic_init_mat = self.logic_matrices()

        self.logic_result = self.get_operator(
            self.logic_operators[0],
            self.logic_init_mat[:, 0],
            self.logic_init_mat[:, 1],
        )

        for i, lopt in enumerate(self.logic_operators[1:]):
            self.logic_result = self.get_operator(
                lopt, self.logic_result, self.logic_init_mat[:, i + 2]
            )

        # print(self.logic_result.astype(dtype=bool))

        # print(b)
        X = self.logic_init_mat
        y = self.logic_result.astype(dtype=int).reshape(len(self.logic_result), 1)
        self.logic_res_mat = np.concatenate((X, y), axis=1)
        self.export()

    def export(self):
        """export the random-generated truth-tables as csv.
        
        The header inculdes all x-variables like x_0, x_1, ... and y as truht or false-value.
        The footer includes the logical expression like: `or(x_5, or(x_4, ...`. The function
        goes always from the inner to the outer ring. As result of the bijektivity, capsuled
        expression can be generated by re-wirrting the expression by switching the variables.
        
        Notes
        -----
        
        If `self.fname == None` the csv-name is based on the math-expression of the logical
        operators exported to an hashlib-value.
        
        """

        header = ""
        for i in range(self.logic_variables):
            header += "x_{},  ".format(str(i))
        header += "  y"
        # Generate math expression as footer
        # Start with the inner-function
        exprs = self.logic_operators[0] + "(x_0, x_1)"
        # Adding the the rest of the logicial expressions
        for i, item in enumerate(self.logic_operators[1:]):
            exprs = "{}(x_{}, {})".format(item, str(i + 2), exprs)
        # Generate the export-name
        if not self.fname:
            self.fname = "{}.csv".format(
                hashlib.sha256(exprs.encode("utf-8")).hexdigest()
            )
        # Export
        np.savetxt(
            self.fname,
            self.logic_res_mat,
            delimiter=",",
            header=header,
            footer=exprs,
            fmt="%5i",
        )


if __name__ == "__main__":
    # print(LogicOperators().lg_and([1,0], [1, 1]))
    LogicGenerator(logic_items=1).generator()
