from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

class MatrixOperations:
    def __init__(self):
        self.matrices = {}

    def input_matrix(self, matrix_name, matrix_data):
        try:
            matrix_data = np.array(matrix_data).tolist()
            self.matrices[matrix_name] = matrix_data
            return f"Matrix {matrix_name} saved successfully."
        except Exception as e:
            raise ValueError(f"Invalid matrix format: {str(e)}")

    def matrix_multiplication(self, mat1_name, mat2_name):
        X = self.matrices[mat1_name]
        Y = self.matrices[mat2_name]

        result = np.dot(X, Y).tolist()
        return result

    def matrix_addition(self, mat1_name, mat2_name):
        X = self.matrices[mat1_name]
        Y = self.matrices[mat2_name]

        result = (np.array(X) + np.array(Y)).tolist()
        return result

    def matrix_subtraction(self, mat1_name, mat2_name):
        X = self.matrices[mat1_name]
        Y = self.matrices[mat2_name]

        result = (np.array(X) - np.array(Y)).tolist()
        return result

    def matrix_inversion(self, mat1_name):
        matrix = np.array(self.matrices[mat1_name])
        if np.linalg.det(matrix) == 0:
            return [f"Inverse does not exist for Matrix {mat1_name}"]
        inv = np.linalg.inv(matrix).tolist()
        return inv

    def matrix_determinant(self, mat1_name):
        matrix = np.array(self.matrices[mat1_name])
        det = np.linalg.det(matrix)
        return [f"The determinant is: {det} unit(s)"]

    def matrix_transpose(self, mat1_name):
        matrix = self.matrices[mat1_name]
        transpose = np.transpose(matrix).tolist()
        return transpose
    
    def matrix_adjoint(self, mat1_name):
        matrix = np.array(self.matrices[mat1_name])
        adjoint = np.matrix.getH(matrix)
        return adjoint.tolist()

    def matrix_eigen(self, mat1_name):
        matrix = np.array(self.matrices[mat1_name])
        eigvals, _ = np.linalg.eig(matrix)
        return [f"Eigen values for Matrix {mat1_name}: {eigvals.tolist()}"]

    def matrix_scalar_multiplication(self, mat1_name, scalar):
        matrix = np.array(self.matrices[mat1_name])
        result = (matrix * scalar).tolist()
        return result
        
matrix_operations = MatrixOperations()

@app.route('/')
def index():
    return render_template('index.html', matrices=matrix_operations.matrices)

@app.route('/add_matrix', methods=['POST'])
def add_matrix():
    matrix_name = request.form['matrix_name']
    matrix_data_str = request.form['matrix_data']
    
    try:
        # Parse the JSON string into a list of lists
        matrix_list = json.loads(matrix_data_str)
        
        # Convert to float
        matrix_list = [[float(cell) for cell in row] for row in matrix_list]
        
        message = matrix_operations.input_matrix(matrix_name, matrix_list)
        flash(message)
    except Exception as e:
        flash(f"Error adding matrix: {str(e)}")
    
    return redirect(url_for('index'))

@app.route('/save_result', methods=['POST'])
def save_result():
    try:
        result_data = request.form['result_data']
        result_name = request.form['result_name']
        
        # Use json.loads instead of eval for safety
        import json
        result_matrix = json.loads(result_data)
        
        # Convert to proper matrix format if needed
        if not isinstance(result_matrix[0], list):
            result_matrix = [result_matrix]  # Make it 2D if it's 1D
            
        # Save the result
        matrix_operations.input_matrix(result_name, result_matrix)
        flash(f"Result saved as '{result_name}' successfully.")
    except Exception as e:
        flash(f"Error saving result: {str(e)}")
    
    return redirect(url_for('index'))

@app.route('/perform_operation', methods=['POST'])
def perform_operation():
    operation = request.form['operation']
    mat1_name = request.form.get('mat1_name')
    mat2_name = request.form.get('mat2_name')
    scalar_value = request.form.get('scalar_value')

    # Validate matrix existence for the first matrix
    if mat1_name not in matrix_operations.matrices:
        flash(f"Matrix '{mat1_name}' does not exist.")
        return redirect(url_for('index'))

    # Validate second matrix for operations requiring two matrices
    if operation in ['add', 'subtract', 'multiply'] and mat2_name not in matrix_operations.matrices:
        return render_template('index.html', matrices=matrix_operations.matrices, result=[f"Error: Two matrices need to be entered for this operation"])

    try:
        if operation == 'multiply':
            result = matrix_operations.matrix_multiplication(mat1_name, mat2_name)
            if not isinstance(result, list):
                result = result.tolist()
        elif operation == 'add':
            result = matrix_operations.matrix_addition(mat1_name, mat2_name)
            if not isinstance(result, list):
                result = result.tolist()
        elif operation == 'subtract':
            result = matrix_operations.matrix_subtraction(mat1_name, mat2_name)
            if not isinstance(result, list):
                result = result.tolist()
        elif operation == 'invert':
            result = matrix_operations.matrix_inversion(mat1_name)
            if not isinstance(result, list):
                result = result.tolist()
        elif operation == 'determinant':
            result = matrix_operations.matrix_determinant(mat1_name)
        elif operation == 'transpose':
            result = matrix_operations.matrix_transpose(mat1_name)
            if not isinstance(result, list):
                result = result.tolist()
        elif operation == 'eigen':
            result = matrix_operations.matrix_eigen(mat1_name)
        elif operation == 'scalar_multiply':  # Handle scalar multiplication
            scalar = float(scalar_value)
            result = matrix_operations.matrix_scalar_multiplication(mat1_name, scalar)
        elif operation == 'adjoint':
            result = matrix_operations.matrix_adjoint(mat1_name)
            if not isinstance(result, list):
                result = result.tolist()
        else:
            result = "Invalid operation."

        if not isinstance(result, list):
            result = [result]

        return render_template('index.html', 
                               matrices=matrix_operations.matrices, 
                               result=result, 
                               operation=operation)
    except KeyError as e:
        result = [f"KeyError: The matrix '{mat1_name}' does not exist."]
    except Exception as e:
        result = [f"An error occurred: {e}"]

    return render_template('index.html', matrices=matrix_operations.matrices, result=result)


if __name__ == '__main__':
    app.run(debug=True)