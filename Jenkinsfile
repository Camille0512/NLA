properties([
    pipelineTriggers([githubPush()])
])

pipeline {
    agent any
    tools {
        git 'Default'  // Uses the Git tool configured in Jenkins Global Tools
    }
    triggers {
        pollSCM('H/5 * * * *')  // Check every 5 mins
    }
    environment {
        GIT_BRANCH = 'refs/heads/develop'
        GIT_URL = 'https://github.com/Camille0512/NLA.git'
        JENKINS_CREDENTIAL_ID = 'jenkins_nla'
        JENKINS_LOG = '/Users/camilleli/Programs/NLA_new/NLA'
        DATETIME = sh(script: 'date +"%Y%m%d_%H%M%S"', returnStdout: true).trim()
    }
    stages {
        stage('Checkout PR') {
            steps {
                sh 'echo "Start PR checkout"'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: GIT_BRANCH]], // ${sha1}
                    extensions: [
                        [$class: 'CloneOption', depth: 1, shallow: true]
                    ],
                    userRemoteConfigs: [[
                        url: GIT_URL,
                        credentialsId: JENKINS_CREDENTIAL_ID,
                        refspec: '+refs/pull/*:refs/remotes/origin/pr/*'
                    ]]
                ])
                sh 'echo "Finish PR checkout"'
            }
        }
        stage('Build') {
            steps {
                sh 'echo "Start Build"'
                sh '''
                    python3 -m venv venv
                    source venv/bin/activate
                    python3 -m pip install --upgrade pip
                    pip3 install -r requirements.txt
                '''
                sh 'echo "Finish Build"'
            }
        }
        stage('Test') {
            parallel {
                stage('Sample Test') {
                    steps {
                        sh 'echo "Start Sample Test"'
                        sh '''
                            source venv/bin/activate
                            python3 -m pytest --junitxml=${JENKINS_LOG}/JenkinsLogs/surefire-reports/${DATETIME}_sample_test-results.xml
                        '''
//                         junit '${JENKINS_LOG}/JenkinsLogs/surefire-reports/${DATETIME}_sample_test-results.xml'
                        sh 'echo "Finish Sample Test"'
                    }
                }
                stage('LU Decomposition Test') {
                    steps {
                        sh 'echo "Start LU Decomposition Test"'
                        sh 'ls -la ${JENKINS_LOG}/JenkinsLogs/surefire-reports/'
                        sh '''
                            source venv/bin/activate
                            python3 -m pytest --junitxml=${JENKINS_LOG}/JenkinsLogs/surefire-reports/${DATETIME}_lu_decomposition_test-results.xml
                        '''
//                         junit '${JENKINS_LOG}/JenkinsLogs/surefire-reports/${DATETIME}_lu_decomposition_test-results.xml'
                        sh 'echo "Finish LU Decomposition Test"'
                    }
                }
            }
        }
        stage('Deploy') {
            steps {
                echo "🚀 Deploying..."
            }
        }
    }
}

node {
    cleanWs()  // Clean workspace after build
}